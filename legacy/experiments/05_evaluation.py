"""
Phase 1 Re-Gate: budget-matched quality and throughput evaluation.

Replaces the original feasibility experiment's "ParamMap vs random" comparison
with the one that actually decides whether domain conditioning works: ParamMap
vs a domain-agnostic importance ranking at an identical block budget. Also
measures real tokens/s, which no earlier experiment did.
"""

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from paramap.eval import (
    ParamMapEvaluator,
    check_gates,
    format_gate_report,
    format_quality_table,
    format_throughput_table,
)
from paramap.mapper import ParameterMap

MAP_PATH = "maps/qwen2.5_0.5b_param_map.safetensors"
BUDGET_RATIOS = (0.5, 0.3)


def main() -> int:
    if not os.path.exists(MAP_PATH):
        print(f"Parameter map not found: {MAP_PATH}")
        print("Run experiments/02_feasibility_experiment.py first to profile and build it.")
        return 1

    param_map = ParameterMap.load(MAP_PATH)
    model_name = param_map.metadata.model_name_or_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print("        PARAMAP PHASE 1 RE-GATE: BUDGET-MATCHED EVALUATION")
    print(f"        Model: {model_name} | Device: {device}")
    print(f"        Map: {param_map.num_layers}L x {param_map.num_blocks}B, "
          f"domains={param_map.metadata.domain_ids}")
    print("=" * 78)

    print("\n[1/3] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    model.eval()
    if device.type != "cuda":
        model.to(device)

    evaluator = ParamMapEvaluator(model, tokenizer, param_map, device=device)

    print("\n[2/3] Quality: perplexity on held-out in-domain text")
    print("      (strategies are budget-matched to exactly k blocks per layer)\n")
    quality_rows = evaluator.evaluate_quality(budget_ratios=BUDGET_RATIOS)

    print("\n[3/3] Throughput: tokens/s, masked vs genuinely sliced\n")
    throughput_rows = evaluator.evaluate_throughput(budget_ratios=BUDGET_RATIOS)

    print("\n" + "=" * 78)
    print("QUALITY\n")
    print(format_quality_table(quality_rows))
    print("\nTHROUGHPUT\n")
    print(format_throughput_table(throughput_rows))

    for budget_ratio in BUDGET_RATIOS:
        print()
        print(format_gate_report(check_gates(quality_rows, throughput_rows, budget_ratio=budget_ratio)))

    return 0


if __name__ == "__main__":
    sys.exit(main())
