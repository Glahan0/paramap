"""
Measurement for budgeted residency.

The claim being tested is narrow and checkable: lowering the VRAM budget must
reduce VRAM actually used, must not change what the model says, and costs
generation speed in proportion to how much has to cross the bus.

So every run reports three things together -- peak VRAM, tokens/s, and whether
the generated token ids are identical to the all-in-VRAM baseline. A speed or
memory number without the identity check would not mean anything.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .residency import (
    ResidencyPlan,
    StreamingModel,
    format_bytes,
    plan_residency,
)

DEFAULT_PROMPT = (
    "Explain in a few sentences why keeping only part of a language model's "
    "weights in GPU memory can still produce exactly the same output."
)


@dataclass
class RunResult:
    label: str
    tokens_per_s: float
    vram_peak_bytes: int
    generated_ids: List[int]
    text: str
    plan: Optional[ResidencyPlan] = None
    identical_to_baseline: Optional[bool] = None
    transfer_bytes: int = 0
    # Leading generated tokens equal to the baseline's (all of them when identical).
    matching_tokens: Optional[int] = None

    @property
    def vram_peak_gb(self) -> float:
        return self.vram_peak_bytes / 1024**3


@torch.no_grad()
def run_generation(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 64,
    warmup_runs: int = 1,
    repeats: int = 3,
) -> Tuple[List[int], str, float, int]:
    """
    Greedy, length-pinned generation. Returns (token_ids, text, tokens/s, peak
    VRAM bytes).

    Greedy decoding is what makes the identity check meaningful: any difference
    in the arithmetic would show up as a different token.
    """
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = encoded["input_ids"].shape[1]
    use_cuda = device.type == "cuda" and torch.cuda.is_available()

    def run() -> torch.Tensor:
        return model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    for _ in range(warmup_runs):
        run()
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    rates: List[float] = []
    output: Optional[torch.Tensor] = None
    for _ in range(repeats):
        start = time.perf_counter()
        output = run()
        if use_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if elapsed > 0:
            rates.append(max_new_tokens / elapsed)

    peak = torch.cuda.max_memory_allocated(device) if use_cuda else 0
    rates.sort()
    tokens_per_s = rates[len(rates) // 2] if rates else float("nan")

    generated = output[0][prompt_length:].tolist() if output is not None else []
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return generated, text, tokens_per_s, peak


def benchmark_budgets(
    model: nn.Module,
    tokenizer: Any,
    budgets: Sequence[Any],
    *,
    device: Optional[torch.device] = None,
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 64,
    max_context_tokens: int = 2048,
    staging_slots: int = 2,
    bandwidth_gb_s: Optional[float] = None,
    mode: str = "exact",
    prefill_chunk_tokens: Any = "auto",
    pin_reserve: Any = "2GB",
    verbose: bool = True,
) -> List[RunResult]:
    """
    Runs the model once per VRAM budget and once as an all-in-VRAM baseline,
    reusing a single model instance (each budget re-places the weights).

    The baseline goes through exactly the same code path with an unconstrained
    budget, so it is not a separately-written fast path that could diverge.

    In mode="fast" CPU-computed layers round differently from the GPU, so a
    budget's output may legitimately differ; `matching_tokens` says how far it
    agreed with the baseline.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: List[RunResult] = []

    baseline_plan = plan_residency(
        model,
        "1024GB",
        max_context_tokens=max_context_tokens,
        staging_slots=staging_slots,
        bandwidth_gb_s=bandwidth_gb_s or 0.0,
    )
    baseline: Optional[RunResult] = None
    needed = (
        baseline_plan.vram_weight_bytes
        + baseline_plan.kv_cache_reserve_bytes
        + baseline_plan.activation_reserve_bytes
    )
    free = torch.cuda.mem_get_info(device)[0] if device.type == "cuda" else None
    if free is not None and needed > free:
        # A model larger than VRAM has no all-in-VRAM run to compare with.
        # Streaming runs the same kernels on the same bytes at every budget,
        # so exact budgets must still agree with each other.
        if verbose:
            print(
                f"Baseline skipped: the whole model needs {format_bytes(needed)} of VRAM, "
                f"{format_bytes(free)} is free. Output is compared with the first budget instead."
            )
    else:
        if verbose:
            print("Baseline: every weight in VRAM")
        with StreamingModel(model, baseline_plan, device=device):
            ids, text, tps, peak = run_generation(
                model, tokenizer, prompt, device, max_new_tokens=max_new_tokens
            )
        baseline = RunResult(
            label="baseline (all in VRAM)",
            tokens_per_s=tps,
            vram_peak_bytes=peak,
            generated_ids=ids,
            text=text,
            plan=baseline_plan,
            identical_to_baseline=True,
        )
        results.append(baseline)
        if verbose:
            print(f"  {tps:6.1f} tok/s | peak VRAM {format_bytes(peak)}")

    for budget in budgets:
        if verbose:
            print(f"\nBudget: {budget}")
        plan = plan_residency(
            model,
            budget,
            max_context_tokens=max_context_tokens,
            staging_slots=staging_slots,
            bandwidth_gb_s=bandwidth_gb_s,
            mode=mode,
            prefill_chunk_tokens=prefill_chunk_tokens,
        )
        if verbose:
            cpu_text = f" / {len(plan.cpu_layers)} on CPU" if plan.cpu_layers else ""
            print(
                f"  plan: {len(plan.resident_layers)} resident / "
                f"{len(plan.streamed_layers)} streamed{cpu_text}, "
                f"weights in VRAM {format_bytes(plan.vram_weight_bytes)}"
            )
            if plan.ram_warning():
                print(plan.ram_warning())

        with StreamingModel(model, plan, device=device, pin_reserve=pin_reserve) as streaming:
            ids, text, tps, peak = run_generation(
                model, tokenizer, prompt, device, max_new_tokens=max_new_tokens
            )
            transferred = streaming.transfer_bytes_total
            pinned_layers = streaming.stats()["pinned_layers"]

        reference = baseline.generated_ids if baseline else ids
        identical = ids == reference
        matching = next(
            (i for i, (a, b) in enumerate(zip(ids, reference)) if a != b),
            min(len(ids), len(reference)),
        )
        result = RunResult(
            label=str(budget) if baseline else f"{budget} (reference)",
            tokens_per_s=tps,
            vram_peak_bytes=peak,
            generated_ids=ids,
            text=text,
            plan=plan,
            identical_to_baseline=identical,
            transfer_bytes=transferred,
            matching_tokens=matching,
        )
        results.append(result)
        if baseline is None:
            baseline = result
        if verbose:
            if identical:
                verdict = "identical"
            elif plan.is_exact:
                verdict = "DIVERGED"
            else:
                verdict = f"same for first {matching}/{len(ids)} tokens (CPU rounding)"
            pinned_text = ""
            if plan.streamed_layers and pinned_layers < len(plan.streamed_layers):
                pinned_text = (
                    f" | only {pinned_layers}/{len(plan.streamed_layers)} streamed layers "
                    "pinned (host RAM low), so slower than the bus allows"
                )
            print(
                f"  {tps:6.1f} tok/s | peak VRAM {format_bytes(peak)} | output {verdict}"
                f"{pinned_text}"
            )

    return results


def format_results(results: Sequence[RunResult]) -> str:
    baseline = results[0] if results else None
    header = (
        f"{'Configuration':<24} | {'GPU/stream/CPU':>14} | {'Peak VRAM':>11} | "
        f"{'VRAM saved':>11} | {'tokens/s':>9} | {'vs base':>8} | {'Output':>9}"
    )
    lines = [header, "-" * len(header)]

    for result in results:
        plan = result.plan
        layers = (
            f"{len(plan.resident_layers)}/{len(plan.streamed_layers)}/{len(plan.cpu_layers)}"
            if plan
            else "-"
        )
        if baseline and baseline.vram_peak_bytes and result is not baseline:
            saved = baseline.vram_peak_bytes - result.vram_peak_bytes
            saved_text = f"{100.0 * saved / baseline.vram_peak_bytes:.1f}%"
        else:
            saved_text = "--"
        if baseline and baseline.tokens_per_s and result is not baseline:
            speed_text = f"{100.0 * (result.tokens_per_s / baseline.tokens_per_s - 1.0):+.1f}%"
        else:
            speed_text = "--"
        if result.identical_to_baseline is None:
            output_text = "?"
        elif result.identical_to_baseline:
            output_text = "identical"
        elif plan is not None and not plan.is_exact:
            output_text = f"{result.matching_tokens}/{len(result.generated_ids)} same"
        else:
            output_text = "DIVERGED"

        lines.append(
            f"{result.label:<24} | {layers:>14} | {format_bytes(result.vram_peak_bytes):>11} | "
            f"{saved_text:>11} | {result.tokens_per_s:>9.1f} | {speed_text:>8} | {output_text:>9}"
        )

    return "\n".join(lines)
