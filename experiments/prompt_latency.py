"""
Time to first token and decode speed by prompt length, per mode.

Prompts are the start of WikiText-2 (test), cut to each length. "first token"
is the prompt pass alone; decode is the median rate over the following tokens.

    python experiments/prompt_latency.py --budget 2GB --lengths 25,512,2000
    python experiments/prompt_latency.py --budget 800MB   # no slot: prompts on the CPU
"""

import argparse
import time

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from paramap.residency import InsufficientVRAMError, StreamingModel, plan_residency


@torch.no_grad()
def measure(model, ids, device, decode_steps):
    cache = DynamicCache()
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = model(ids.to(device), past_key_values=cache, use_cache=True)
    token = out.logits[0, -1].argmax().view(1, 1)
    torch.cuda.synchronize()
    first = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(decode_steps):
        token = model(token.to(device), past_key_values=cache, use_cache=True).logits[0, -1].argmax().view(1, 1)
    torch.cuda.synchronize()
    return first, decode_steps / (time.perf_counter() - start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--budget", default="2GB")
    parser.add_argument("--lengths", default="25,512,2000")
    parser.add_argument("--decode-steps", type=int, default=32)
    args = parser.parse_args()

    import pyarrow.parquet as pq

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).eval()
    path = hf_hub_download(
        "Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset"
    )
    text = "".join(pq.read_table(path).column("text").to_pylist()[:400])
    all_ids = tokenizer(text, return_tensors="pt").input_ids
    lengths = [int(x) for x in args.lengths.split(",")]

    for label, kwargs in [
        ("exact", dict(mode="exact")),
        ("fast", dict(mode="fast")),
        ("fast, prompts on CPU", dict(mode="fast")),
    ]:
        try:
            plan = plan_residency(model, args.budget, max_context_tokens=max(lengths) + 64, **kwargs)
        except InsufficientVRAMError:
            print(f"{label:<22} does not fit in {args.budget}", flush=True)
            continue
        if label.endswith("on CPU"):
            if not plan.cpu_layers:
                continue
            plan.staging_slots = 0  # the behaviour before prompts got the slot
        split = f"{len(plan.resident_layers)}/{len(plan.streamed_layers)}/{len(plan.cpu_layers)}"
        with StreamingModel(model, plan, device=device):
            measure(model, all_ids[:, :8], device, 2)  # warm-up
            for n in lengths:
                first, rate = measure(model, all_ids[:, :n], device, args.decode_steps)
                print(f"{label:<22} {split:>8} GPU/stream/CPU | prompt {n:5d} tokens: "
                      f"first token {first:7.2f} s | decode {rate:5.2f} tok/s", flush=True)


if __name__ == "__main__":
    main()
