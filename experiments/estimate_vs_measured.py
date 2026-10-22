"""
The fast planner's per-token estimate against measured generate speed.

For each budget and mode: the plan, its estimate (decoder layers only) and
the tokens/s run_generation measures. The difference between the two is the
per-token cost the estimate leaves out (output head, generate's own work).

    python experiments/estimate_vs_measured.py --model Qwen/Qwen2.5-3B-Instruct --budgets 5GB,2GB
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from paramap.benchmark import DEFAULT_PROMPT, run_generation
from paramap.residency import (
    StreamingModel,
    measure_cpu_layer_ms,
    measure_gpu_layer_ms,
    plan_residency,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--budgets", default="5GB,2GB")
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    lines = []

    def say(text=""):
        print(text, flush=True)
        lines.append(text)

    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    cpu_ms, gpu_ms = measure_cpu_layer_ms(model), measure_gpu_layer_ms(model)
    say(f"{args.model} | {torch.cuda.get_device_name(0)} | {time.strftime('%Y-%m-%d %H:%M')}")
    say(f"standalone: cpu layer {cpu_ms:.2f} ms, gpu layer {gpu_ms:.2f} ms")
    say()
    say(f"{'budget':>7} | {'mode':>5} | {'GPU/stream/CPU':>14} | {'est ms':>7} | {'meas ms':>7} | {'diff':>6} | tok/s")
    for budget in [b for b in args.budgets.split(",") if b]:
        for mode in ("fast", "exact-stream"):
            plan = plan_residency(
                model, budget, mode="fast" if mode == "fast" else "exact",
                cpu_layer_ms=cpu_ms, gpu_layer_ms=gpu_ms, max_context_tokens=2048,
            )
            if mode != "fast":
                # The exact plan carries no timings; give it the same ones.
                plan.gpu_layer_ms, plan.cpu_layer_ms = gpu_ms, 0.0
            with StreamingModel(model, plan, device=device):
                _, _, tps, _ = run_generation(model, tokenizer, DEFAULT_PROMPT, device, max_new_tokens=args.tokens)
            est = plan.estimated_seconds_per_token * 1e3
            meas = 1e3 / tps
            split = f"{len(plan.resident_layers)}/{len(plan.streamed_layers)}/{len(plan.cpu_layers)}"
            say(f"{budget:>7} | {mode[:5]:>5} | {split:>14} | {est:>7.1f} | {meas:>7.1f} | {meas - est:>+6.1f} | {tps:.1f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
