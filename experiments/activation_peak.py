"""
How much VRAM does a forward pass need beyond the weights and the KV cache?

The planner reserves `estimate_activation_reserve_bytes` for it. It was a
guess (2% of the weights plus 16 logits rows); this measures the real
transient peak against prompt length, and the estimate is now sized from it.

For each prompt length: VRAM allocated before generate (weights, staging
slots), the peak during a generate of a few tokens, and the KV cache the
planner would reserve for that length. transient = peak - before - KV.

    python experiments/activation_peak.py --model Qwen/Qwen2.5-3B-Instruct
    python experiments/activation_peak.py --model Qwen/Qwen2.5-7B-Instruct --budget 7GB --mmap
"""

import argparse
import time

import torch

from paramap.residency import (
    StreamingModel,
    estimate_activation_reserve_bytes,
    estimate_kv_cache_bytes,
    format_bytes,
    plan_residency,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--budget", default="1024GB")
    parser.add_argument("--mode", default="exact")
    parser.add_argument("--mmap", action="store_true", help="map weights from disk (large models)")
    parser.add_argument("--lengths", default="16,128,512,1024,2048,4096")
    parser.add_argument("--new-tokens", type=int, default=4)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    lines = []

    def say(text: str = "") -> None:
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
    config = model.config
    lengths = [int(x) for x in args.lengths.split(",") if x]

    plan = plan_residency(
        model, args.budget, mode=args.mode, max_context_tokens=max(lengths) + args.new_tokens
    )
    say(f"{args.model} | {torch.cuda.get_device_name(0)} | {time.strftime('%Y-%m-%d %H:%M')}")
    say(
        f"budget {args.budget} ({args.mode}): {len(plan.resident_layers)} resident / "
        f"{len(plan.streamed_layers)} streamed / {len(plan.cpu_layers)} cpu | "
        f"planned reserve {format_bytes(plan.activation_reserve_bytes)}"
    )
    say(
        f"hidden {config.hidden_size}, intermediate {config.intermediate_size}, "
        f"heads {config.num_attention_heads}/{config.num_key_value_heads}, "
        f"vocab {config.vocab_size}, attn {getattr(config, '_attn_implementation', '?')}"
    )
    say()
    say(f"{'tokens':>7} | {'before':>10} | {'peak':>10} | {'KV est':>10} | {'transient':>10} | {'estimate':>10} | per token")

    with torch.no_grad(), StreamingModel(model, plan, device=device):
        for length in lengths:
            ids = torch.randint(100, 20000, (1, length), device=device)
            mask = torch.ones_like(ids)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            before = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_new_tokens=args.new_tokens,
                min_new_tokens=args.new_tokens,
                do_sample=False,
                pad_token_id=0,
            )
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated(device)
            kv = estimate_kv_cache_bytes(config, length + args.new_tokens, 2)
            transient = peak - before - kv
            estimate = estimate_activation_reserve_bytes(model, config, 2, length)
            say(
                f"{length:>7} | {format_bytes(before):>10} | {format_bytes(peak):>10} | "
                f"{format_bytes(kv):>10} | {format_bytes(transient):>10} | {format_bytes(estimate):>10} | "
                f"{transient / length / 1024:.1f} KB"
            )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
