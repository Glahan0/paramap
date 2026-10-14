"""
Evaluation harness for ParamMap.

Answers the three questions the project's claims actually rest on:

1. **Quality.** What does a block budget cost in perplexity, and does
   domain-conditioned selection beat a *domain-agnostic importance ranking* at
   the same budget? Beating random pruning proves only that importance-based
   selection works at all -- any sane ranking does that. The importance
   baseline here is the control that PLAN section 7 / Plan B item 3 describes;
   if ParamMap cannot beat it on in-domain text, domain conditioning is not
   what is buying the gain.

2. **Throughput.** Does running fewer blocks actually emit tokens faster?
   Measured two ways, because they are not the same thing:
     - `masked`: zero out inactive neurons (what the demos and the web UI do).
       Every weight stays resident and every matmul keeps its original shape,
       so this frees no memory and can only add overhead.
     - `sliced`: physically rebuild each FFN from just the active neurons'
       weights. This is what a real runtime would do and the only variant that
       can show a speed-up.

3. **Gates.** The PLAN section 9 targets: <= 5% in-domain perplexity loss, and
   throughput that improves or regresses by at most 10%.

Every selection strategy is budget-matched: each returns *exactly* k blocks per
layer, so any difference comes from which blocks were picked, not how many.
"""

import contextlib
import math
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..arch.base import ModelArchAdapter
from ..arch.qwen import QwenArchitectureAdapter
from .mapper import ParameterMap
from .runtime import BlockSparseMLP

# ---------------------------------------------------------------------------
# Held-out evaluation texts
# ---------------------------------------------------------------------------
# These must NOT overlap the seed prompts in domains.py: the parameter map is
# built from those, so scoring on them would measure the map against its own
# training data.

HELD_OUT_EVAL_TEXTS: Dict[str, List[str]] = {
    "math_formal": [
        "Determine whether the series sum of 1/(n*log n) from n=2 to infinity converges, and justify the test you used.",
        "Bir ABC ucgeninde kenarlar a=7, b=9 ve aradaki aci 60 derece ise ucuncu kenari kosinus teoremiyle bulun.",
        "Show that every finite integral domain is a field.",
    ],
    "code_python_system": [
        "Explain why a Python generator holding an open file handle can leak descriptors when the consumer breaks out of the loop early.",
        "Bir Django projesinde N+1 sorgu problemini select_related ve prefetch_related ile nasil cozersiniz?",
        "Write a context manager that retries a flaky HTTP call with exponential backoff and jitter.",
    ],
    "general_conversation": [
        "I have been feeling a bit overwhelmed at work lately and I am not sure how to bring it up with my manager.",
        "Bu yaz tatilinde ailemle birlikte gidebilecegimiz, cocuklar icin de uygun sakin bir yer onerir misin?",
        "What is the most underrated small kitchen gadget and why do you think people overlook it?",
    ],
    "creative_writing": [
        "The clockmaker kept one watch that ran backwards, and he never explained to anyone what it was counting down to.",
        "Issiz bir tren istasyonunda, her gece ayni saatte gelen ama hic durmayan bir trenin hikayesini anlat.",
        "Write the opening paragraph of a novel in which the narrator slowly reveals they are unreliable.",
    ],
}

DEFAULT_THROUGHPUT_PROMPT = (
    "Explain, step by step, how the memory bandwidth of a GPU limits the speed "
    "of autoregressive token generation in a large language model."
)


# ---------------------------------------------------------------------------
# Budget-matched block selection strategies
# ---------------------------------------------------------------------------

def _topk_sorted(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k indices of a 1-D score vector, returned in ascending index order."""
    return torch.topk(scores, k=k).indices.sort().values


def select_paramap(param_map: ParameterMap, domain_idx: int, k: int) -> List[torch.Tensor]:
    """Domain-conditioned: the k blocks most selective for `domain_idx`."""
    return [
        _topk_sorted(param_map.block_scores[layer, :, domain_idx], k)
        for layer in range(param_map.num_layers)
    ]


def select_paramap_with_core(
    param_map: ParameterMap,
    domain_idx: int,
    k: int,
    core_fraction: float = 0.2,
) -> List[torch.Tensor]:
    """
    The scheduler's actual policy, held to an exact budget: reserve part of k for
    domain-agnostic 'core' blocks, then fill the rest by domain selectivity.
    `BlockScheduler` unions the two sets instead, which can overshoot the budget;
    here the total is exactly k so the comparison stays fair.
    """
    core_k = max(1, min(k, int(round(k * core_fraction))))
    selected: List[torch.Tensor] = []

    for layer in range(param_map.num_layers):
        core = _topk_sorted(param_map.block_scores[layer].mean(dim=-1), core_k)
        core_set = set(core.tolist())

        ranked = torch.argsort(param_map.block_scores[layer, :, domain_idx], descending=True)
        fill = [b for b in ranked.tolist() if b not in core_set][: k - core_k]

        blocks = sorted(core_set.union(fill))
        selected.append(torch.tensor(blocks, dtype=torch.long))

    return selected


def select_importance(param_map: ParameterMap, k: int) -> List[torch.Tensor]:
    """
    Domain-agnostic control: the k blocks with the highest mean score across all
    domains, i.e. 'just keep the generally important parameters'. This is the
    baseline that decides whether domain conditioning earns its complexity.
    """
    return [
        _topk_sorted(param_map.block_scores[layer].mean(dim=-1), k)
        for layer in range(param_map.num_layers)
    ]


def select_random(param_map: ParameterMap, k: int, seed: int = 42) -> List[torch.Tensor]:
    """Random k blocks per layer -- the weak reference baseline."""
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randperm(param_map.num_blocks, generator=generator)[:k].sort().values
        for _ in range(param_map.num_layers)
    ]


def blocks_to_neurons(param_map: ParameterMap, layer_idx: int, blocks: torch.Tensor) -> torch.Tensor:
    """Maps clustered block indices to the physical neuron rows they cover."""
    if blocks.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    order = param_map.neuron_order[layer_idx]
    block_size = param_map.block_size
    return torch.cat([order[b * block_size : (b + 1) * block_size] for b in blocks.tolist()])


def blocks_for_budget(param_map: ParameterMap, budget_ratio: float) -> int:
    """Number of blocks per layer for a budget ratio (at least 1)."""
    return max(1, int(round(param_map.num_blocks * budget_ratio)))


# ---------------------------------------------------------------------------
# Applying a selection to a live model
# ---------------------------------------------------------------------------

def _mask_hook(mask: torch.Tensor):
    def hook(module, inputs):
        return (inputs[0] * mask.view(1, 1, -1),)
    return hook


@contextlib.contextmanager
def masked_blocks(
    model: nn.Module,
    param_map: ParameterMap,
    blocks_per_layer: Sequence[torch.Tensor],
    adapter: Optional[ModelArchAdapter] = None,
):
    """
    Zeroes inactive neurons at each FFN's down-projection input. Mathematically
    identical to dropping them (down_proj is a linear sum over neurons) but frees
    no memory and shrinks no matmul -- useful for quality measurement only.
    """
    adapter = adapter or QwenArchitectureAdapter()
    handles = []
    try:
        for layer_idx, blocks in enumerate(blocks_per_layer):
            mlp = adapter.get_layer_mlp(model, layer_idx)
            weight = mlp.down_proj.weight
            mask = torch.zeros(weight.shape[1], dtype=weight.dtype, device=weight.device)
            neurons = blocks_to_neurons(param_map, layer_idx, blocks).to(weight.device)
            if neurons.numel() > 0:
                mask[neurons] = 1.0
            handles.append(mlp.down_proj.register_forward_pre_hook(_mask_hook(mask)))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextlib.contextmanager
def sliced_blocks(
    model: nn.Module,
    param_map: ParameterMap,
    blocks_per_layer: Sequence[torch.Tensor],
    adapter: Optional[ModelArchAdapter] = None,
):
    """
    Replaces each FFN with a BlockSparseMLP built from only the active neurons'
    weights, so the matmuls really do get smaller. Originals are restored on exit.
    """
    adapter = adapter or QwenArchitectureAdapter()
    hidden_size = adapter.get_hidden_size(model)
    intermediate_size = adapter.get_intermediate_size(model)

    act_fn = "silu"
    config = getattr(model, "config", None)
    if config is not None:
        act_fn = (
            getattr(config, "hidden_act", None)
            or getattr(config, "hidden_activation", None)
            or "silu"
        )

    saved: Dict[int, nn.Module] = {}
    try:
        for layer_idx, blocks in enumerate(blocks_per_layer):
            original = adapter.get_layer_mlp(model, layer_idx)
            saved[layer_idx] = original

            sparse = BlockSparseMLP(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                block_size=param_map.block_size,
                act_fn=act_fn,
            )
            sparse.load_active_blocks(
                original.gate_proj.weight.data,
                original.up_proj.weight.data,
                original.down_proj.weight.data,
                blocks,
                param_map.neuron_order[layer_idx],
                original.down_proj.weight.device,
            )
            adapter.set_layer_mlp(model, layer_idx, sparse)
        yield
    finally:
        for layer_idx, original in saved.items():
            adapter.set_layer_mlp(model, layer_idx, original)


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_perplexity(
    model: nn.Module,
    tokenizer: Any,
    texts: Iterable[str],
    device: torch.device,
) -> Tuple[float, int]:
    """
    Token-weighted perplexity over `texts`. Each text is tokenized on its own, so
    no padding position ever enters the loss. Returns (perplexity, token_count).
    """
    total_nll = 0.0
    total_tokens = 0

    for text in texts:
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            continue
        loss = float(model(input_ids=input_ids, labels=input_ids).loss)
        num_predicted = input_ids.shape[1] - 1
        if not math.isfinite(loss):
            return float("inf"), total_tokens + num_predicted
        total_nll += loss * num_predicted
        total_tokens += num_predicted

    if total_tokens == 0:
        return float("nan"), 0

    mean_nll = total_nll / total_tokens
    return (math.exp(mean_nll) if mean_nll < 700 else float("inf")), total_tokens


@torch.no_grad()
def measure_throughput(
    model: nn.Module,
    tokenizer: Any,
    prompt: str = DEFAULT_THROUGHPUT_PROMPT,
    device: Optional[torch.device] = None,
    max_new_tokens: int = 64,
    warmup_runs: int = 3,
    repeats: int = 3,
) -> float:
    """
    Decode throughput in tokens/s. Greedy and length-pinned (`min_new_tokens ==
    max_new_tokens`) so a degraded model cannot look fast by stopping early.
    Returns the median of `repeats` timed runs.
    """
    device = device or next(model.parameters()).device
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

    rates: List[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = run()
        if use_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        generated = output.shape[1] - prompt_length
        if elapsed > 0 and generated > 0:
            rates.append(generated / elapsed)

    return statistics.median(rates) if rates else float("nan")


# ---------------------------------------------------------------------------
# Result rows
# ---------------------------------------------------------------------------

@dataclass
class QualityRow:
    domain_id: str
    strategy: str
    budget_ratio: float
    blocks_per_layer: int
    perplexity: float
    ppl_delta_percent: float  # vs the unmasked model, on the same texts


@dataclass
class ThroughputRow:
    mode: str  # "full" | "masked" | "sliced"
    budget_ratio: float
    blocks_per_layer: int
    tokens_per_s: float
    delta_percent: float  # vs the full model


SELECTION_STRATEGIES = ("paramap", "paramap_core", "importance", "random")


class ParamMapEvaluator:
    """
    Runs budget-matched quality and throughput comparisons for a parameter map
    against the model it was profiled from.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        param_map: ParameterMap,
        device: Optional[torch.device] = None,
        adapter: Optional[ModelArchAdapter] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.param_map = param_map
        self.device = device or next(model.parameters()).device
        self.adapter = adapter or QwenArchitectureAdapter()
        self.throughput_drift_percent: float = float("nan")

        model_layers = self.adapter.get_num_layers(model)
        if model_layers != param_map.num_layers:
            raise ValueError(
                f"Map has {param_map.num_layers} layers but the model has {model_layers}; "
                "they were not profiled from the same architecture."
            )

    # -- selection ---------------------------------------------------------

    def select(self, strategy: str, domain_id: str, k: int) -> List[torch.Tensor]:
        if strategy == "importance":
            return select_importance(self.param_map, k)
        if strategy == "random":
            return select_random(self.param_map, k)

        try:
            domain_idx = self.param_map.metadata.domain_ids.index(domain_id)
        except ValueError:
            raise ValueError(
                f"Domain '{domain_id}' is not in this map ({self.param_map.metadata.domain_ids}). "
                "A map loaded from disk must carry its real domain ids."
            )

        if strategy == "paramap":
            return select_paramap(self.param_map, domain_idx, k)
        if strategy == "paramap_core":
            return select_paramap_with_core(self.param_map, domain_idx, k)
        raise ValueError(f"Unknown strategy '{strategy}'. Known: {SELECTION_STRATEGIES}")

    # -- quality -----------------------------------------------------------

    def evaluate_quality(
        self,
        eval_texts: Optional[Dict[str, List[str]]] = None,
        budget_ratios: Sequence[float] = (0.5,),
        strategies: Sequence[str] = SELECTION_STRATEGIES,
        verbose: bool = True,
    ) -> List[QualityRow]:
        """
        For each domain, measures perplexity on held-out in-domain text under each
        strategy at each budget, plus the unmasked baseline.
        """
        eval_texts = eval_texts or HELD_OUT_EVAL_TEXTS
        rows: List[QualityRow] = []

        for domain_id, texts in eval_texts.items():
            if domain_id not in self.param_map.metadata.domain_ids:
                continue

            baseline_ppl, _ = measure_perplexity(self.model, self.tokenizer, texts, self.device)
            rows.append(
                QualityRow(domain_id, "full", 1.0, self.param_map.num_blocks, baseline_ppl, 0.0)
            )
            if verbose:
                print(f"  [{domain_id}] full-model perplexity: {baseline_ppl:.3f}")

            for budget_ratio in budget_ratios:
                k = blocks_for_budget(self.param_map, budget_ratio)
                for strategy in strategies:
                    blocks = self.select(strategy, domain_id, k)
                    with masked_blocks(self.model, self.param_map, blocks, self.adapter):
                        ppl, _ = measure_perplexity(self.model, self.tokenizer, texts, self.device)
                    delta = ((ppl / baseline_ppl) - 1.0) * 100.0 if math.isfinite(ppl) else float("inf")
                    rows.append(QualityRow(domain_id, strategy, budget_ratio, k, ppl, delta))
                    if verbose:
                        print(
                            f"    {strategy:<13} budget={budget_ratio:.0%} "
                            f"k={k:<3} ppl={ppl:>10.3f}  ({delta:+.1f}%)"
                        )

        return rows

    # -- throughput --------------------------------------------------------

    def evaluate_throughput(
        self,
        prompt: str = DEFAULT_THROUGHPUT_PROMPT,
        domain_id: Optional[str] = None,
        budget_ratios: Sequence[float] = (0.5, 0.3),
        max_new_tokens: int = 64,
        verbose: bool = True,
    ) -> List[ThroughputRow]:
        """
        Compares full-model throughput against both the masked and the genuinely
        sliced variants at each budget.
        """
        domain_id = domain_id or self.param_map.metadata.domain_ids[0]
        rows: List[ThroughputRow] = []

        # The full model is measured twice, before and after the variants. GPU
        # clocks ramp up under sustained load, so a single leading baseline makes
        # everything measured after it look faster. The two readings bracket that
        # drift; `throughput_drift_percent` reports it and the deltas below are
        # taken against their mean.
        full_first = measure_throughput(
            self.model, self.tokenizer, prompt, self.device, max_new_tokens=max_new_tokens
        )
        if verbose:
            print(f"  full model (first) : {full_first:.1f} tok/s")

        variant_rows: List[ThroughputRow] = []
        for budget_ratio in budget_ratios:
            k = blocks_for_budget(self.param_map, budget_ratio)
            blocks = self.select("paramap_core", domain_id, k)

            for mode, applier in (("masked", masked_blocks), ("sliced", sliced_blocks)):
                with applier(self.model, self.param_map, blocks, self.adapter):
                    tps = measure_throughput(
                        self.model,
                        self.tokenizer,
                        prompt,
                        self.device,
                        max_new_tokens=max_new_tokens,
                    )
                variant_rows.append(ThroughputRow(mode, budget_ratio, k, tps, float("nan")))
                if verbose:
                    print(f"  {mode:<7} budget={budget_ratio:.0%} k={k:<3} {tps:>7.1f} tok/s")

        full_second = measure_throughput(
            self.model, self.tokenizer, prompt, self.device, max_new_tokens=max_new_tokens
        )
        if verbose:
            print(f"  full model (second): {full_second:.1f} tok/s")

        reference = statistics.fmean([full_first, full_second])
        drift = ((full_second / full_first) - 1.0) * 100.0 if full_first else float("nan")

        rows.append(ThroughputRow("full", 1.0, self.param_map.num_blocks, reference, 0.0))
        for row in variant_rows:
            row.delta_percent = ((row.tokens_per_s / reference) - 1.0) * 100.0 if reference else float("nan")
            rows.append(row)

        self.throughput_drift_percent = drift
        if verbose:
            print(f"  baseline drift between the two full-model runs: {drift:+.1f}%")
            if abs(drift) > 5.0:
                print(
                    "  WARNING: the full-model baseline moved by more than 5% during the run, "
                    "so differences smaller than that are measurement noise, not sparsity."
                )

        return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_quality_table(rows: Sequence[QualityRow]) -> str:
    header = (
        f"{'Domain':<22} | {'Strategy':<13} | {'Budget':>6} | {'k':>4} | "
        f"{'Perplexity':>12} | {'vs full':>9}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        ppl = f"{row.perplexity:.3f}" if math.isfinite(row.perplexity) else "inf"
        delta = "--" if row.strategy == "full" else f"{row.ppl_delta_percent:+.1f}%"
        lines.append(
            f"{row.domain_id:<22} | {row.strategy:<13} | {row.budget_ratio:>5.0%} | "
            f"{row.blocks_per_layer:>4} | {ppl:>12} | {delta:>9}"
        )
    return "\n".join(lines)


def format_throughput_table(rows: Sequence[ThroughputRow]) -> str:
    header = f"{'Mode':<10} | {'Budget':>6} | {'k':>4} | {'tokens/s':>10} | {'vs full':>9}"
    lines = [header, "-" * len(header)]
    for row in rows:
        delta = "--" if row.mode == "full" else f"{row.delta_percent:+.1f}%"
        lines.append(
            f"{row.mode:<10} | {row.budget_ratio:>5.0%} | {row.blocks_per_layer:>4} | "
            f"{row.tokens_per_s:>10.1f} | {delta:>9}"
        )
    return "\n".join(lines)


def check_gates(
    quality_rows: Sequence[QualityRow],
    throughput_rows: Sequence[ThroughputRow],
    budget_ratio: float = 0.5,
    max_ppl_loss_percent: float = 5.0,
    max_throughput_loss_percent: float = 10.0,
) -> Dict[str, Any]:
    """
    Evaluates the PLAN section 9 targets, plus the one comparison that decides
    whether domain conditioning is doing any work.
    """
    def at_budget(strategy: str) -> List[QualityRow]:
        return [
            r for r in quality_rows
            if r.strategy == strategy and abs(r.budget_ratio - budget_ratio) < 1e-9
        ]

    paramap_rows = at_budget("paramap_core") or at_budget("paramap")
    importance_rows = at_budget("importance")
    random_rows = at_budget("random")

    def mean_delta(rows: Sequence[QualityRow]) -> float:
        finite = [r.ppl_delta_percent for r in rows if math.isfinite(r.ppl_delta_percent)]
        return statistics.fmean(finite) if finite else float("inf")

    paramap_delta = mean_delta(paramap_rows)
    importance_delta = mean_delta(importance_rows)
    random_delta = mean_delta(random_rows)

    # Per-domain head-to-head, which a mean can hide.
    importance_by_domain = {r.domain_id: r.perplexity for r in importance_rows}
    wins = sum(
        1 for r in paramap_rows
        if r.domain_id in importance_by_domain and r.perplexity < importance_by_domain[r.domain_id]
    )

    def delta_for(mode: str) -> float:
        matching = [
            r for r in throughput_rows
            if r.mode == mode and abs(r.budget_ratio - budget_ratio) < 1e-9
        ]
        return matching[0].delta_percent if matching else float("nan")

    sliced_delta = delta_for("sliced")
    masked_delta = delta_for("masked")

    return {
        "budget_ratio": budget_ratio,
        "quality_gate_target_percent": max_ppl_loss_percent,
        "paramap_mean_ppl_loss_percent": paramap_delta,
        "quality_gate_passed": paramap_delta <= max_ppl_loss_percent,
        "importance_mean_ppl_loss_percent": importance_delta,
        "random_mean_ppl_loss_percent": random_delta,
        "beats_random": paramap_delta < random_delta,
        "beats_importance_baseline": paramap_delta < importance_delta,
        "domains_won_vs_importance": f"{wins}/{len(paramap_rows)}",
        "sliced_throughput_delta_percent": sliced_delta,
        "masked_throughput_delta_percent": masked_delta,
        "throughput_gate_passed": (
            math.isfinite(sliced_delta) and sliced_delta >= -max_throughput_loss_percent
        ),
    }


def format_gate_report(gates: Dict[str, Any]) -> str:
    def verdict(passed: bool) -> str:
        return "PASS" if passed else "FAIL"

    return "\n".join([
        f"Gate report at {gates['budget_ratio']:.0%} block budget",
        "-" * 62,
        f"  In-domain perplexity loss (target <= {gates['quality_gate_target_percent']:.0f}%)",
        f"    ParamMap            : {gates['paramap_mean_ppl_loss_percent']:+.1f}%"
        f"   [{verdict(gates['quality_gate_passed'])}]",
        f"    Importance baseline : {gates['importance_mean_ppl_loss_percent']:+.1f}%",
        f"    Random baseline     : {gates['random_mean_ppl_loss_percent']:+.1f}%",
        "",
        f"  Beats random pruning      : {verdict(gates['beats_random'])}",
        f"  Beats importance baseline : {verdict(gates['beats_importance_baseline'])}"
        f"   (per-domain wins: {gates['domains_won_vs_importance']})",
        "",
        "  Throughput vs full model (target >= -10%)",
        f"    sliced (real sparsity) : {gates['sliced_throughput_delta_percent']:+.1f}%"
        f"   [{verdict(gates['throughput_gate_passed'])}]",
        f"    masked (simulation)    : {gates['masked_throughput_delta_percent']:+.1f}%",
        "-" * 62,
    ])
