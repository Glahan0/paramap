"""
Does fast mode (layers computed on the CPU) cost any quality?

Compares each configuration against the all-GPU baseline on WikiText-2 (test):

  prefill -- 32 chunks x 512 tokens, teacher-forced in one forward pass each:
             perplexity, KL(baseline || config) per token, top-1 agreement
  decode  -- 8 x 256 tokens fed one at a time through the KV cache, the path
             generation actually takes (CPU decode uses different kernels
             from CPU prefill)

For scale, it also measures a configuration that stays entirely on the GPU and
only swaps the attention kernel (sdpa -> eager): a numerical difference nobody
would call a quality change.

    python experiments/perplexity_fast_vs_exact.py [--chunks 32] [--model ...]

Each configuration's row is printed as soon as it finishes, and the baseline's
logits are kept on disk (--cache-dir, ~5 GB for the defaults), so an
interrupted run loses nothing finished and can resume with --reuse-baseline.

Needs pyarrow to read the dataset (pip install pyarrow).
"""

import argparse
import math
import os
import tempfile
import time

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from paramap.residency import StreamingModel, plan_residency


def load_wikitext_ids(tokenizer, num_tokens):
    import pyarrow.parquet as pq

    path = hf_hub_download(
        "Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset"
    )
    text = "".join(pq.read_table(path).column("text").to_pylist())
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    assert ids.numel() >= num_tokens, "dataset shorter than requested"
    return ids[:num_tokens]


@torch.no_grad()
def prefill_logits(model, chunk, device):
    return model(chunk.unsqueeze(0).to(device), use_cache=False).logits[0, :-1]


@torch.no_grad()
def decode_logits(model, seq, device):
    cache = DynamicCache()
    rows = []
    for t in range(seq.numel() - 1):
        out = model(seq[t : t + 1].view(1, 1).to(device), past_key_values=cache, use_cache=True)
        rows.append(out.logits[0, -1])
    return torch.stack(rows)


def compare(base, other, targets):
    """Sums over tokens: (nll_base, nll_other, kl, top1_agree, n, bit_identical)."""
    base, other = base.float(), other.float()
    lp_base = F.log_softmax(base, dim=-1)
    lp_other = F.log_softmax(other, dim=-1)
    t = targets.to(base.device).view(-1, 1)
    nll_base = -lp_base.gather(1, t).sum().item()
    nll_other = -lp_other.gather(1, t).sum().item()
    kl = (lp_base.exp() * (lp_base - lp_other)).sum().item()
    agree = (base.argmax(-1) == other.argmax(-1)).sum().item()
    return nll_base, nll_other, kl, agree, targets.numel(), torch.equal(base, other)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--chunks", type=int, default=32)
    parser.add_argument("--chunk-len", type=int, default=512)
    parser.add_argument("--decode-seqs", type=int, default=8)
    parser.add_argument("--decode-len", type=int, default=256)
    parser.add_argument("--cache-dir", default=None, help="where baseline logits are kept")
    parser.add_argument("--reuse-baseline", action="store_true",
                        help="skip the baseline run if its logits are already in --cache-dir")
    parser.add_argument("--skip-exact", action="store_true",
                        help="skip the exact-mode sanity check (slowest configuration)")
    parser.add_argument("--only", default=None,
                        help="comma-separated label prefixes to run, e.g. 'baseline,fast 0.8GB'")
    args = parser.parse_args()

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).eval()
    ids = load_wikitext_ids(tokenizer, args.chunks * args.chunk_len)
    chunks = ids.view(args.chunks, args.chunk_len)
    decode_seqs = [chunks[i, : args.decode_len] for i in range(args.decode_seqs)]
    cache_dir = args.cache_dir or tempfile.mkdtemp(prefix="paramap_ppl_")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"{args.model}: {ids.numel()} prefill tokens, "
          f"{args.decode_seqs}x{args.decode_len} decode tokens; baseline logits in {cache_dir}")

    common = dict(max_context_tokens=2048, bandwidth_gb_s=None)
    configs = [
        ("baseline (all GPU)", lambda: plan_residency(model, "1024GB", **common), None),
        ("GPU, eager attention", lambda: plan_residency(model, "1024GB", **common), "eager"),
        ("exact 2GB (stream)", lambda: plan_residency(model, "2GB", mode="exact", **common), None),
        ("fast 2GB (CPU)", lambda: plan_residency(model, "2GB", mode="fast", **common), None),
        ("fast 0.8GB (CPU)", lambda: plan_residency(model, "0.8GB", mode="fast", **common), None),
    ]
    if args.skip_exact:
        configs = [c for c in configs if not c[0].startswith("exact")]
    if args.only:
        prefixes = [p.strip() for p in args.only.split(",") if p.strip()]
        configs = [c for c in configs if any(c[0].startswith(p) for p in prefixes)]

    def base_path(kind, i):
        return os.path.join(cache_dir, f"{kind}{i}.pt")

    have_baseline = all(
        os.path.exists(base_path("chunk", i)) for i in range(args.chunks)
    ) and all(os.path.exists(base_path("decode", j)) for j in range(len(decode_seqs)))
    if args.reuse_baseline and have_baseline:
        print("  baseline (all GPU): reusing logits from --cache-dir", flush=True)
        configs = configs[1:]

    print()
    print(f"{'Configuration':<22} | {'GPU/str/CPU':>11} | {'PPL':>7} | {'dPPL':>8} | "
          f"{'KL nats/tok':>11} | {'top-1':>7} | {'dec dPPL':>8} | {'decode KL':>10} | "
          f"{'dec top-1':>9} | bit-identical | time", flush=True)

    for label, make_plan, attn in configs:
        is_base = label.startswith("baseline")
        plan = make_plan()
        if attn:
            model.set_attn_implementation(attn)
        start = time.perf_counter()
        t = [0.0, 0.0, 0.0, 0, 0, True]
        d = [0.0, 0.0, 0.0, 0, 0, True]
        with StreamingModel(model, plan, device=device):
            for i in range(args.chunks):
                logits = prefill_logits(model, chunks[i], device)
                if is_base:
                    torch.save(logits.cpu(), base_path("chunk", i))
                    continue
                r = compare(torch.load(base_path("chunk", i)).to(device), logits, chunks[i, 1:])
                t = [a + b for a, b in zip(t[:5], r[:5])] + [t[5] and r[5]]
            for j, seq in enumerate(decode_seqs):
                logits = decode_logits(model, seq, device)
                if is_base:
                    torch.save(logits.cpu(), base_path("decode", j))
                    continue
                r = compare(torch.load(base_path("decode", j)).to(device), logits, seq[1:])
                d = [a + b for a, b in zip(d[:5], r[:5])] + [d[5] and r[5]]
        if attn:
            model.set_attn_implementation("sdpa")

        split = f"{len(plan.resident_layers)}/{len(plan.streamed_layers)}/{len(plan.cpu_layers)}"
        elapsed = f"{time.perf_counter() - start:.0f}s"
        if is_base:
            print(f"{label:<22} | {split:>11} | {'(ref)':>7} | {'--':>8} | {'--':>11} | {'--':>7} | "
                  f"{'--':>8} | {'--':>10} | {'--':>9} | -- | {elapsed}", flush=True)
            continue
        base_ppl, ppl = math.exp(t[0] / t[4]), math.exp(t[1] / t[4])
        print(f"{label:<22} | {split:>11} | {ppl:7.3f} | {100 * (ppl / base_ppl - 1):+7.3f}% | "
              f"{t[2] / t[4]:11.2e} | {100 * t[3] / t[4]:6.2f}% | "
              f"{100 * (math.exp((d[1] - d[0]) / d[4]) - 1):+7.3f}% | {d[2] / d[4]:10.2e} | "
              f"{100 * d[3] / d[4]:8.2f}% | prefill {t[5]}, decode {d[5]} | {elapsed}"
              f"  [baseline PPL {base_ppl:.3f}, decode PPL {math.exp(d[0] / d[4]):.3f}]",
              flush=True)

if __name__ == "__main__":
    main()
