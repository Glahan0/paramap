"""
What does one CPU-computed layer really cost per decoded token?

The fast planner times a decoder layer on its own (`measure_cpu_layer_ms`) and
scales it by _CPU_IN_DECODE_FACTOR (1.5) for what real decode adds. That factor
came from two models; this measures it directly.

Decode time per token is measured with k of the model's layers on the CPU, for
several k. Its slope over k is the cost of moving one more layer from the GPU
to the CPU, free of the per-token cost that does not depend on k (embedding,
output head, sampling). slope + gpu_layer_ms = one CPU layer in decode.

    python experiments/cpu_layer_cost.py --model Qwen/Qwen2.5-3B-Instruct --cpu-layers 4,12,20,28
    python experiments/cpu_layer_cost.py --model Qwen/Qwen2.5-7B-Instruct --mmap --cpu-layers 4,8,12,16
"""

import argparse
import statistics
import time

import torch

from paramap.residency import (
    StreamingModel,
    measure_cpu_layer_ms,
    measure_gpu_layer_ms,
    plan_residency,
)


@torch.no_grad()
def decode_ms_per_token(model, device, prompt_tokens, steps, warmup=4):
    from transformers import DynamicCache

    cache = DynamicCache()
    ids = torch.randint(100, 20000, (1, prompt_tokens), device=device)
    token = model(ids, past_key_values=cache, use_cache=True).logits[0, -1].argmax().view(1, 1)
    times = []
    for step in range(warmup + steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        token = model(token, past_key_values=cache, use_cache=True).logits[0, -1].argmax().view(1, 1)
        torch.cuda.synchronize()
        if step >= warmup:
            times.append((time.perf_counter() - start) * 1e3)
    return statistics.median(times)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--mmap", action="store_true")
    parser.add_argument("--cpu-layers", default="4,12,20,28")
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    lines = []

    def say(text=""):
        print(text, flush=True)
        lines.append(text)

    device = torch.device("cuda")
    if args.mmap:
        from paramap.loader import load_model_mmap

        model, _ = load_model_mmap(args.model, torch.bfloat16)
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval()

    cpu_ms = measure_cpu_layer_ms(model)
    gpu_ms = measure_gpu_layer_ms(model)
    say(f"{args.model} | {torch.cuda.get_device_name(0)} | {time.strftime('%Y-%m-%d %H:%M')}")
    say(f"standalone: cpu layer {cpu_ms:.2f} ms, gpu layer {gpu_ms:.2f} ms")
    say()
    say(f"{'cpu layers':>10} | {'ms/token':>9}")

    num_layers = len(model.model.layers)
    points = []
    for k in [int(x) for x in args.cpu_layers.split(",") if x]:
        plan = plan_residency(
            model, "1024GB", mode="fast", cpu_layer_ms=cpu_ms, gpu_layer_ms=gpu_ms,
            bandwidth_gb_s=10.0, prefill_chunk_tokens=0,
        )
        plan.resident_layers = list(range(num_layers - k))
        plan.cpu_layers = list(range(num_layers - k, num_layers))
        plan.streamed_layers = []
        plan.staging_slots = 1
        with StreamingModel(model, plan, device=device):
            ms = decode_ms_per_token(model, device, args.prompt_tokens, args.steps)
        points.append((k, ms))
        say(f"{k:>10} | {ms:>9.1f}")

    n = len(points)
    mean_k = sum(k for k, _ in points) / n
    mean_ms = sum(ms for _, ms in points) / n
    slope = sum((k - mean_k) * (ms - mean_ms) for k, ms in points) / sum(
        (k - mean_k) ** 2 for k, _ in points
    )
    intercept = mean_ms - slope * mean_k
    in_decode = slope + gpu_ms
    say()
    say(f"slope {slope:.2f} ms per CPU layer, intercept {intercept:.1f} ms (all on GPU)")
    say(f"CPU layer in decode: {in_decode:.2f} ms = {in_decode / cpu_ms:.2f} x standalone "
        f"(+{in_decode - cpu_ms:.2f} ms)")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
