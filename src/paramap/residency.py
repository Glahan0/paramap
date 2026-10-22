"""
Budgeted, lossless VRAM/RAM residency for transformer decoders.

Every parameter is kept. What changes is *where* it lives at any given moment:
a budgeted set of decoder layers stays permanently in VRAM, and the rest are
held in host RAM and streamed in over PCIe just before they are needed, then
their VRAM slot is reused by the next layer. Arithmetic is unchanged, so the
logits are identical to running the model fully on the GPU -- what you trade is
generation speed for VRAM headroom.

Why this works at all: every decoder layer of a given model has the same
parameter shapes, so a small number of interchangeable GPU "slots" can be
rotated through all streamed layers. Two slots are enough to overlap the
transfer of the next layer with the computation of the current one.

The cost is predictable rather than mysterious. Per generated token the runtime
must move `bytes_streamed_per_token` across the bus, so

    seconds_per_token >= bytes_streamed_per_token / pcie_bandwidth

and `ResidencyPlan` reports that floor before you run anything.

mode="fast" adds a second way to handle a layer that does not fit: compute it
on the CPU, straight from RAM. On a machine whose RAM outreads its PCIe link
that is quicker and needs no VRAM at all, but CPU kernels round differently,
so the output is no longer bit-identical. The library defaults to "exact".
"""

import math
import mmap
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .arch.base import ModelArchAdapter
from .arch.qwen import QwenArchitectureAdapter
from .loader import is_file_backed, release_file_pages

_BYTE_UNITS = {
    "b": 1,
    "kb": 1024,
    "k": 1024,
    "mb": 1024**2,
    "m": 1024**2,
    "gb": 1024**3,
    "g": 1024**3,
    "tb": 1024**4,
    "t": 1024**4,
}


def parse_bytes(value: Any) -> int:
    """
    Accepts an int (bytes) or a human string such as "4GB", "512 MB", "3.5g".
    """
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return int(value)

    text = str(value).strip().lower().replace(" ", "")
    match = re.fullmatch(r"([0-9]*\.?[0-9]+)([a-z]*)", text)
    if not match:
        raise ValueError(f"Could not parse a byte size from {value!r}")

    amount, unit = match.groups()
    multiplier = _BYTE_UNITS.get(unit or "b")
    if multiplier is None:
        raise ValueError(f"Unknown size unit {unit!r} in {value!r}")
    return int(float(amount) * multiplier)


def format_bytes(num_bytes: float) -> str:
    if num_bytes >= 1024**3:
        return f"{num_bytes / 1024**3:.2f} GB"
    if num_bytes >= 1024**2:
        return f"{num_bytes / 1024**2:.1f} MB"
    return f"{num_bytes / 1024:.0f} KB"


# ---------------------------------------------------------------------------
# Measuring the model
# ---------------------------------------------------------------------------

def _tensor_bytes(tensors) -> int:
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


def module_bytes(module: nn.Module, include_buffers: bool = True) -> int:
    total = _tensor_bytes(module.parameters())
    if include_buffers:
        total += _tensor_bytes(module.buffers())
    return total


# Measured with experiments/activation_peak.py on Qwen2.5-0.5B/3B/7B (RTX 3060,
# sdpa): a generate needs ~13 MB plus 3.5-3.7 x intermediate_size elements per
# prompt token beyond the weights and KV cache -- the MLP's gate, up and product
# rows for the whole prompt. The constants round both up.
_ACTIVATION_FIXED_BYTES = 32 * 1024**2
_ACTIVATION_ROWS_PER_TOKEN = 4

# Prompt chunk sizes fast mode chooses from, largest first (0 = whole prompt).
# Processing a prompt in chunks caps that scratch at one chunk's worth
# (Qwen2.5-3B, 4000-token prompt: peak 461 -> 192 MB with the KV cache), but
# attention over chunks rounds differently -- greedy output there matched
# unchunked for 23 of 32 tokens -- so exact mode keeps whole prompts unless
# asked. Smaller chunks also cost prompt time: every chunk copies each CPU
# layer into the GPU slot again (7B, 4096 tokens: 4.7 s whole, 9.9 s at 512).
_FAST_PREFILL_CHUNKS = (0, 2048, 1024, 512)


def estimate_activation_reserve_bytes(
    model: nn.Module,
    config: Any,
    dtype_bytes: int = 2,
    max_context_tokens: int = 4096,
) -> int:
    """
    Headroom to leave for the transient tensors of a forward pass.

    Decode activations are negligible; the peak is the prompt pass, whose MLP
    scratch grows linearly with prompt length (a prompt can be as long as the
    context). Sized from measurement: on Qwen2.5-7B a 4096-token prompt needs
    558 MB, and the earlier guess of 2% of the weights (295 MB) let the peak
    pass a 7 GB budget.

    Covers `generate`, which computes logits for the last position only; a
    plain forward over a long sequence also holds its full logits
    (tokens x vocabulary in fp32) and needs more.
    """
    intermediate = getattr(config, "intermediate_size", None) if config is not None else None
    if not intermediate:
        # No MLP width to go on: fall back to a fraction of the weights.
        return max(64 * 1024**2, int(0.02 * module_bytes(model)))
    per_token = _ACTIVATION_ROWS_PER_TOKEN * intermediate * dtype_bytes
    return _ACTIVATION_FIXED_BYTES + max_context_tokens * per_token


def estimate_kv_cache_bytes(config: Any, max_context_tokens: int, dtype_bytes: int = 2) -> int:
    """
    Bytes the attention cache needs for `max_context_tokens` of context.

    Uses the key/value head count, so grouped-query models are not overcharged.
    Returns 0 when the config does not expose enough information to be honest
    about it -- a wrong reservation is worse than none.
    """
    num_layers = getattr(config, "num_hidden_layers", None)
    num_heads = getattr(config, "num_attention_heads", None)
    hidden_size = getattr(config, "hidden_size", None)
    if not num_layers or not num_heads or not hidden_size:
        return 0

    num_kv_heads = getattr(config, "num_key_value_heads", None) or num_heads
    head_dim = getattr(config, "head_dim", None) or (hidden_size // num_heads)

    # 2 tensors (keys and values) per layer.
    return 2 * num_layers * num_kv_heads * head_dim * max_context_tokens * dtype_bytes


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

@dataclass
class ResidencyPlan:
    """
    How a model's weights will be distributed for a given VRAM budget.

    `anchor_bytes` covers everything outside the decoder stack -- embeddings,
    the final norm, the output head -- which is touched on every token and so is
    never streamed. It lives in VRAM, except the input embedding when it is not
    shared with the output head (`cpu_anchor_bytes`): a lookup reads only the
    rows of the tokens at hand, so it stays on the CPU (see `plan_residency`).

    Each decoder layer is in exactly one of three places:
      resident -- in VRAM for the whole run
      streamed -- in host RAM, copied into a GPU staging slot before it runs
      cpu      -- in host RAM and computed there ("fast" mode only); its
                  hidden state crosses the bus instead of its weights

    Only "exact" plans (no cpu layers) guarantee output identical to running
    every layer on the GPU: CPU kernels round differently.
    """

    num_layers: int
    layer_bytes: int
    anchor_bytes: int
    vram_budget_bytes: int
    kv_cache_reserve_bytes: int
    activation_reserve_bytes: int
    staging_slots: int
    resident_layers: List[int]
    streamed_layers: List[int]
    bandwidth_gb_s: float
    dtype_bytes: int = 2
    cpu_layers: List[int] = field(default_factory=list)
    mode: str = "exact"
    # Per-layer, per-token costs the plan was chosen with (0 when unknown).
    cpu_layer_ms: float = 0.0
    gpu_layer_ms: float = 0.0
    # Host RAM the off-GPU layers may occupy: what was available when the plan
    # was made, less a reserve for the rest of the system (0 = unknown).
    host_ram_room_bytes: int = 0
    # Anchors kept on the CPU (the input embedding table); not in anchor_bytes.
    cpu_anchor_bytes: int = 0
    # Prompts run through the model this many tokens at a time (0 = whole).
    prefill_chunk_tokens: int = 0

    @property
    def host_overflow_bytes(self) -> int:
        """
        Off-GPU weight bytes that do not fit in free RAM. Every one of them is
        read on every token, so they are either read from disk again each
        time or held by squeezing other programs out of RAM -- in either case
        slower, and on Qwen2.5-7B at 4 GB free RAM fell to 0.43 GB.
        """
        if not self.host_ram_room_bytes:
            return 0
        return max(0, self.host_weight_bytes - self.host_ram_room_bytes)

    @property
    def total_weight_bytes(self) -> int:
        return self.anchor_bytes + self.cpu_anchor_bytes + self.num_layers * self.layer_bytes

    @property
    def staging_bytes(self) -> int:
        # Streamed layers rotate through the slots every token; CPU layers use
        # one to run multi-token passes (prompts) on the GPU.
        if not (self.streamed_layers or self.cpu_layers):
            return 0
        return self.staging_slots * self.layer_bytes

    @property
    def resident_weight_bytes(self) -> int:
        return self.anchor_bytes + len(self.resident_layers) * self.layer_bytes

    @property
    def vram_weight_bytes(self) -> int:
        """Weight bytes actually held in VRAM, staging buffers included."""
        return self.resident_weight_bytes + self.staging_bytes

    @property
    def host_weight_bytes(self) -> int:
        """Weight bytes kept off the GPU (in RAM, or for a mapped model on disk)."""
        return (len(self.streamed_layers) + len(self.cpu_layers)) * self.layer_bytes

    @property
    def vram_saved_bytes(self) -> int:
        """Against holding every weight in VRAM."""
        return self.total_weight_bytes - self.vram_weight_bytes

    @property
    def vram_saved_percent(self) -> float:
        if not self.total_weight_bytes:
            return 0.0
        return 100.0 * self.vram_saved_bytes / self.total_weight_bytes

    @property
    def bytes_streamed_per_token(self) -> int:
        return len(self.streamed_layers) * self.layer_bytes

    @property
    def pcie_seconds_per_token(self) -> float:
        if not self.bytes_streamed_per_token or self.bandwidth_gb_s <= 0:
            return 0.0
        return self.bytes_streamed_per_token / (self.bandwidth_gb_s * 1024**3)

    @property
    def tokens_per_s_ceiling(self) -> float:
        """
        Upper bound implied by the bus alone: even with compute fully overlapped
        and zero other cost, generation cannot beat this.
        """
        seconds = self.pcie_seconds_per_token
        return float("inf") if seconds <= 0 else 1.0 / seconds

    @property
    def estimated_seconds_per_token(self) -> float:
        """
        Cost-model estimate used to choose between plans; 0 when the per-layer
        timings are unknown. A streamed layer costs its transfer or its compute,
        whichever is longer, since the two overlap. `cpu_layer_ms` is already
        the in-decode estimate. Anchors and the output head are not counted.
        """
        if not self.gpu_layer_ms:
            return 0.0
        transfer_ms = (
            self.layer_bytes / (self.bandwidth_gb_s * 1024**3) * 1e3 if self.bandwidth_gb_s > 0 else 0.0
        )
        ms = (
            len(self.resident_layers) * self.gpu_layer_ms
            + len(self.streamed_layers) * max(transfer_ms, self.gpu_layer_ms)
            + len(self.cpu_layers) * self.cpu_layer_ms
        )
        return ms / 1e3

    @property
    def is_exact(self) -> bool:
        """True when the output is guaranteed identical to all-in-VRAM."""
        return not self.cpu_layers

    @property
    def is_fully_resident(self) -> bool:
        return not self.streamed_layers and not self.cpu_layers

    def describe(self) -> str:
        lines = [
            f"VRAM budget            : {format_bytes(self.vram_budget_bytes)}",
            f"Model weights (total)  : {format_bytes(self.total_weight_bytes)}"
            f"  ({self.num_layers} layers x {format_bytes(self.layer_bytes)}"
            f" + {format_bytes(self.anchor_bytes + self.cpu_anchor_bytes)} anchors)",
            "",
            f"  resident layers      : {len(self.resident_layers)}/{self.num_layers}"
            f"  ({format_bytes(len(self.resident_layers) * self.layer_bytes)})",
            f"  streamed layers      : {len(self.streamed_layers)}/{self.num_layers}"
            f"  ({format_bytes(len(self.streamed_layers) * self.layer_bytes)} off the GPU)",
            f"  CPU-computed layers  : {len(self.cpu_layers)}/{self.num_layers}"
            f"  ({format_bytes(len(self.cpu_layers) * self.layer_bytes)} off the GPU)",
            f"  anchors (always VRAM): {format_bytes(self.anchor_bytes)}"
            + (
                f"  (+ {format_bytes(self.cpu_anchor_bytes)} input embedding on the CPU)"
                if self.cpu_anchor_bytes
                else ""
            ),
            f"  staging slots        : {self.staging_slots}"
            f"  ({format_bytes(self.staging_bytes)})",
            "",
            f"  weights in VRAM      : {format_bytes(self.vram_weight_bytes)}",
            f"  KV cache reserved    : {format_bytes(self.kv_cache_reserve_bytes)}",
            f"  activation reserve   : {format_bytes(self.activation_reserve_bytes)}"
            + (
                f"  (prompts in chunks of {self.prefill_chunk_tokens} tokens)"
                if self.prefill_chunk_tokens
                else ""
            ),
            f"  VRAM saved vs all-in : {format_bytes(self.vram_saved_bytes)}"
            f"  ({self.vram_saved_percent:.1f}%)",
        ]
        if self.is_fully_resident:
            lines += ["", "  Nothing needs streaming at this budget: full speed."]
            return "\n".join(lines)

        if self.streamed_layers:
            lines += [
                "",
                f"  Streamed per token   : {format_bytes(self.bytes_streamed_per_token)}",
                f"  Bus speed            : {self.bandwidth_gb_s:.2f} GB/s",
                f"  Speed ceiling        : {self.tokens_per_s_ceiling:.1f} tokens/s"
                " (bus alone, best case)",
            ]
        if self.estimated_seconds_per_token > 0:
            lines += [
                "",
                f"  Estimated speed      : {1.0 / self.estimated_seconds_per_token:.1f} tokens/s"
                f" (CPU layer {self.cpu_layer_ms:.1f} ms, GPU layer {self.gpu_layer_ms:.1f} ms)",
            ]
        if self.cpu_layers:
            lines += [
                "  Prompt processing    : CPU layers run on the GPU through a staging slot"
                if self.staging_slots
                else "  Prompt processing    : on the CPU (no room for a staging slot; long"
                " prompts will be slow)",
            ]
        if self.host_overflow_bytes:
            lines += ["", *self.ram_warning().splitlines()]
        lines += [
            "",
            "  Output               : identical to all-in-VRAM"
            if self.is_exact
            else "  Output               : NOT bit-identical (CPU layers round differently;"
            " use --exact to guarantee it)",
        ]
        return "\n".join(lines)

    def ram_warning(self) -> str:
        """Empty when the off-GPU layers fit in free RAM."""
        if not self.host_overflow_bytes:
            return ""
        return (
            f"  WARNING: RAM too small. {format_bytes(self.host_weight_bytes)} of layers live off the GPU,\n"
            f"           but only {format_bytes(self.host_ram_room_bytes)} of RAM is free for them. The other\n"
            f"           {format_bytes(self.host_overflow_bytes)} is re-read from disk every token, or other\n"
            "           programs are pushed out of RAM to hold it: expect less than the\n"
            "           estimated speed and a system low on memory. Raise the budget or\n"
            "           close programs."
        )


def untied_embedding_bytes(model: nn.Module) -> int:
    """Size of the input embedding table if it is not the output head's weight."""
    embedding = _untied_embedding(model)
    return module_bytes(embedding, include_buffers=False) if embedding is not None else 0


def _untied_embedding(model: nn.Module) -> Optional[nn.Module]:
    get_in = getattr(model, "get_input_embeddings", None)
    get_out = getattr(model, "get_output_embeddings", None)
    if get_in is None:
        return None
    try:
        embedding = get_in()
        head = get_out() if get_out is not None else None
    except (AttributeError, NotImplementedError):
        return None
    if not isinstance(embedding, nn.Embedding):
        return None
    head_weight = getattr(head, "weight", None)
    if head_weight is not None and head_weight is embedding.weight:
        return None
    return embedding


class InsufficientVRAMError(RuntimeError):
    """Raised when a budget cannot even hold the always-resident parts."""


# RAM left to the rest of the system when judging whether off-GPU layers fit;
# the same default StreamingModel keeps free when pinning.
_HOST_RAM_RESERVE = 2 * 1024**3


def plan_residency(
    model: nn.Module,
    vram_budget: Any,
    *,
    host_ram_available: Any = None,
    embedding_on_cpu: bool = True,
    prefill_chunk_tokens: Any = "auto",
    **kwargs,
) -> ResidencyPlan:
    """
    See `_choose_plan` for how the budget is spent.

    `prefill_chunk_tokens` splits prompts into passes of that many tokens,
    which shrinks the activation reserve from a whole context's worth to one
    chunk's. "auto" chunks in fast mode only: chunked attention rounds
    differently, and exact mode promises the output of an unmodified
    `generate`. 0 (or None) keeps prompts whole.

    With `embedding_on_cpu` the input embedding table stays on the CPU when it
    is not shared with the output head. A lookup is a copy, not arithmetic, so
    the output is unchanged; on Qwen2.5-7B it frees 1.02 GB of VRAM, about two
    decoder layers. A tied table is needed by the output head on the GPU
    anyway, so it stays there.

    Also records how much RAM
    the off-GPU layers have room for (`host_ram_available`, measured when
    omitted, less a 2 GB reserve) so the plan can say when they will not fit.

    Only for a model mapped from its files: an ordinarily loaded model already
    holds every weight in RAM, so free RAM says nothing about whether they fit.
    """
    cpu_anchor_bytes = untied_embedding_bytes(model) if embedding_on_cpu else 0
    if str(prefill_chunk_tokens) == "auto" and kwargs.get("mode", "exact") == "fast":
        plan = _plan_fast_chunked(model, vram_budget, cpu_anchor_bytes, kwargs)
    else:
        chunk = 0 if str(prefill_chunk_tokens) == "auto" else max(0, int(prefill_chunk_tokens or 0))
        plan = _choose_plan(
            model, vram_budget, cpu_anchor_bytes=cpu_anchor_bytes, prefill_chunk_tokens=chunk, **kwargs
        )
        plan.prefill_chunk_tokens = chunk
    plan.cpu_anchor_bytes = cpu_anchor_bytes
    if not is_file_backed(model):
        return plan
    if host_ram_available is None:
        try:
            import psutil

            host_ram_available = psutil.virtual_memory().available
        except ImportError:
            host_ram_available = 0
    available = parse_bytes(host_ram_available)
    plan.host_ram_room_bytes = max(1, available - _HOST_RAM_RESERVE) if available else 0
    return plan


def _plan_fast_chunked(
    model: nn.Module, vram_budget: Any, cpu_anchor_bytes: int, kwargs: Dict[str, Any]
) -> ResidencyPlan:
    """
    Fast mode's "auto" prompt chunking: the largest chunk (whole prompt
    first) that keeps as many layers on the GPU as the smallest one does.
    Chunking only pays when it buys VRAM for a layer; beyond that, bigger
    chunks mean fewer passes over the CPU layers' weights per prompt. On
    Qwen2.5-7B at 7 GB and a 4096-token context that is 1024: 12 GPU layers
    instead of 11, in 4 passes rather than 8.
    """
    kwargs = dict(kwargs)
    adapter = kwargs.get("adapter")
    # Measured once, not once per candidate.
    if kwargs.get("cpu_layer_ms") is None:
        kwargs["cpu_layer_ms"] = measure_cpu_layer_ms(model, adapter)
    if kwargs.get("gpu_layer_ms") is None:
        kwargs["gpu_layer_ms"] = measure_gpu_layer_ms(model, adapter)
    if kwargs.get("bandwidth_gb_s") is None:
        from .device.profiler import DeviceProfiler

        kwargs["bandwidth_gb_s"] = DeviceProfiler().benchmark_bandwidth().get("h2d_gb_s", 0.0)

    context = kwargs.get("max_context_tokens", 4096)
    plans: List[ResidencyPlan] = []
    error: Optional[InsufficientVRAMError] = None
    for chunk in _FAST_PREFILL_CHUNKS:
        if chunk and chunk >= context:
            continue  # no different from the whole prompt
        try:
            plan = _choose_plan(
                model, vram_budget, cpu_anchor_bytes=cpu_anchor_bytes, prefill_chunk_tokens=chunk, **kwargs
            )
        except InsufficientVRAMError as exc:
            error = exc
            continue
        plan.prefill_chunk_tokens = chunk
        plans.append(plan)
    if not plans:
        raise error
    best = max(len(plan.resident_layers) for plan in plans)
    return next(plan for plan in plans if len(plan.resident_layers) == best)


def _choose_plan(
    model: nn.Module,
    vram_budget: Any,
    *,
    max_context_tokens: int = 4096,
    staging_slots: int = 2,
    bandwidth_gb_s: Optional[float] = None,
    activation_reserve: Any = "auto",
    adapter: Optional[ModelArchAdapter] = None,
    mode: str = "exact",
    cpu_layer_ms: Optional[float] = None,
    gpu_layer_ms: Optional[float] = None,
    cpu_anchor_bytes: int = 0,
    prefill_chunk_tokens: int = 0,
) -> ResidencyPlan:
    """
    Decides how many decoder layers fit in `vram_budget` and how many must be
    streamed from host RAM.

    mode="exact" (the default) only streams, so the output is identical to
    running everything on the GPU. mode="fast" may also compute layers on the
    CPU and picks whatever mix the cost model says is quickest; see
    `_plan_fast`. `cpu_layer_ms` / `gpu_layer_ms` override the per-layer
    timings it uses (measured / estimated when omitted).

    The budget is spent in this order, because each item is non-negotiable
    before the next one is worth anything: anchors (touched every token), the KV
    cache for `max_context_tokens`, an activation scratch reserve, the staging
    slots that make streaming possible, and finally as many resident layers as
    the remainder allows.

    Which layers stay resident is cost-neutral -- every layer runs on every
    token -- so the leading layers are kept, which lets the prefetcher run ahead
    during them.
    """
    adapter = adapter or QwenArchitectureAdapter()
    layers = adapter.get_layers(model)
    num_layers = len(layers)
    if num_layers == 0:
        raise ValueError("Model exposes no decoder layers.")

    budget = parse_bytes(vram_budget)

    layer_sizes = [module_bytes(layer) for layer in layers]
    layer_bytes = max(layer_sizes)  # slots must fit the largest layer
    total_model_bytes = module_bytes(model)
    anchor_bytes = max(0, total_model_bytes - sum(layer_sizes) - cpu_anchor_bytes)

    first_param = next(model.parameters(), None)
    dtype_bytes = first_param.element_size() if first_param is not None else 2

    config = getattr(model, "config", None)
    kv_bytes = (
        estimate_kv_cache_bytes(config, max_context_tokens, dtype_bytes) if config is not None else 0
    )
    activation_reserve_bytes = (
        estimate_activation_reserve_bytes(
            model,
            config,
            dtype_bytes,
            # Scratch scales with the tokens in one pass, not the whole context.
            min(max_context_tokens, prefill_chunk_tokens) if prefill_chunk_tokens else max_context_tokens,
        )
        if str(activation_reserve) == "auto"
        else parse_bytes(activation_reserve)
    )

    if bandwidth_gb_s is None:
        from .device.profiler import DeviceProfiler

        bandwidth_gb_s = DeviceProfiler().benchmark_bandwidth().get("h2d_gb_s", 0.0)

    if mode not in ("exact", "fast"):
        raise ValueError(f"mode must be 'exact' or 'fast', not {mode!r}")

    if mode == "fast":
        if cpu_layer_ms is None:
            cpu_layer_ms = measure_cpu_layer_ms(model, adapter)
        if gpu_layer_ms is None:
            gpu_layer_ms = measure_gpu_layer_ms(model, adapter)
        return _plan_fast(
            num_layers=num_layers,
            layer_bytes=layer_bytes,
            anchor_bytes=anchor_bytes,
            budget=budget,
            kv_bytes=kv_bytes,
            activation_reserve_bytes=activation_reserve_bytes,
            max_context_tokens=max_context_tokens,
            staging_slots=staging_slots,
            bandwidth_gb_s=float(bandwidth_gb_s),
            dtype_bytes=dtype_bytes,
            cpu_layer_ms=float(cpu_layer_ms),
            gpu_layer_ms=float(gpu_layer_ms),
        )

    fixed = anchor_bytes + kv_bytes + activation_reserve_bytes
    if fixed >= budget:
        raise InsufficientVRAMError(
            f"A {format_bytes(budget)} budget cannot cover the parts that must stay resident: "
            f"{format_bytes(anchor_bytes)} of anchors + {format_bytes(kv_bytes)} of KV cache for "
            f"{max_context_tokens} tokens + {format_bytes(activation_reserve_bytes)} of activation "
            f"reserve = {format_bytes(fixed)}. Raise the budget, shorten the context, or pick a "
            "smaller model."
        )

    # Can everything simply live in VRAM?
    if fixed + num_layers * layer_bytes <= budget:
        return ResidencyPlan(
            num_layers=num_layers,
            layer_bytes=layer_bytes,
            anchor_bytes=anchor_bytes,
            vram_budget_bytes=budget,
            kv_cache_reserve_bytes=kv_bytes,
            activation_reserve_bytes=activation_reserve_bytes,
            staging_slots=0,
            resident_layers=list(range(num_layers)),
            streamed_layers=[],
            bandwidth_gb_s=float(bandwidth_gb_s),
            dtype_bytes=dtype_bytes,
        )

    slots = max(1, int(staging_slots))
    available = budget - fixed - slots * layer_bytes
    if available < 0:
        # Streaming needs at least one slot; try a single one before giving up.
        slots = 1
        available = budget - fixed - layer_bytes
        if available < 0:
            raise InsufficientVRAMError(
                f"A {format_bytes(budget)} budget leaves no room for even one staging slot "
                f"({format_bytes(layer_bytes)} per layer) after {format_bytes(fixed)} of anchors, "
                "KV cache and activation reserve."
            )

    num_resident = min(num_layers, int(available // layer_bytes))
    # Keep at least one layer streamed, otherwise the slots are wasted.
    if num_resident >= num_layers:
        num_resident = num_layers - 1

    return ResidencyPlan(
        num_layers=num_layers,
        layer_bytes=layer_bytes,
        anchor_bytes=anchor_bytes,
        vram_budget_bytes=budget,
        kv_cache_reserve_bytes=kv_bytes,
        activation_reserve_bytes=activation_reserve_bytes,
        staging_slots=slots,
        resident_layers=list(range(num_resident)),
        streamed_layers=list(range(num_resident, num_layers)),
        bandwidth_gb_s=float(bandwidth_gb_s),
        dtype_bytes=dtype_bytes,
    )


def _plan_fast(
    *,
    num_layers: int,
    layer_bytes: int,
    anchor_bytes: int,
    budget: int,
    kv_bytes: int,
    activation_reserve_bytes: int,
    max_context_tokens: int,
    staging_slots: int,
    bandwidth_gb_s: float,
    dtype_bytes: int,
    cpu_layer_ms: float,
    gpu_layer_ms: float,
) -> ResidencyPlan:
    """
    Chooses between two plans for the layers that do not fit in VRAM and keeps
    whichever has the lower estimated time per token:

      cpu    -- compute them on the CPU. Needs no VRAM at all (not even a
                staging slot), and their KV cache lives in host RAM, so the KV
                reservation only covers the resident layers. RAM bandwidth is
                the limit.
      stream -- the exact plan: stream them. Each costs a PCIe transfer or
                its own GPU compute, whichever is longer (they overlap).

`cpu_layer_ms` is a standalone timing; in real decode a CPU layer also pays
for its hooks and for moving its inputs off the GPU, a fixed cost per layer
added as _CPU_IN_DECODE_MS. And because only the exact plan guarantees identical
output, the CPU plan must be predicted at least _FAST_MIN_SPEEDUP faster to be
chosen: on Qwen2.5-0.5B the two measured within 3% of each other while the CPU
plan's output diverged after 3 tokens, which is all cost and no gain.

    The two are deliberately not mixed. Interleaving streamed layers between
    CPU layers looked like it should overlap transfers with CPU compute, but
    both read the same RAM: measured on Qwen2.5-3B (Ryzen 5 5500, RTX 3060),
    every mixed plan was slower than the pure CPU plan at the same budget
    (2 GB: 3.3 vs ~4 tokens/s), and a cost model that predicted otherwise
    cannot be trusted to pick between them.
    """
    kv_per_layer = kv_bytes / num_layers if num_layers else 0
    fixed = anchor_bytes + activation_reserve_bytes
    if fixed >= budget:
        raise InsufficientVRAMError(
            f"A {format_bytes(budget)} budget cannot cover the parts that must stay resident: "
            f"{format_bytes(anchor_bytes)} of anchors + {format_bytes(activation_reserve_bytes)} "
            f"of activation reserve = {format_bytes(fixed)}. Raise the budget or pick a smaller "
            "model."
        )

    cpu_layer_ms = cpu_layer_ms + _CPU_IN_DECODE_MS

    # cpu: every GPU layer is resident and carries its own KV cache. One
    # staging slot is kept back for prompts: a multi-token pass costs a CPU
    # layer ~3 ms per token (bf16 GEMM without native CPU support), so a
    # 512-token prompt would take ~40 s on Qwen2.5-3B, while streaming the
    # layer through the slot costs ~12 ms whatever the length.
    def cpu_resident(slot_bytes: int) -> int:
        room = budget - fixed - slot_bytes
        return max(0, min(num_layers, int(room // (layer_bytes + kv_per_layer)))) if room >= 0 else -1

    resident, prefill_slots = cpu_resident(0), 0
    if resident < num_layers and cpu_resident(layer_bytes) >= 0:
        resident, prefill_slots = cpu_resident(layer_bytes), 1
    cpu_ms = resident * gpu_layer_ms + (num_layers - resident) * cpu_layer_ms
    best = (resident, 0, prefill_slots)

    # stream: every layer runs on the GPU, so the whole KV cache is reserved.
    if resident < num_layers and bandwidth_gb_s > 0:
        transfer_ms = layer_bytes / (bandwidth_gb_s * 1024**3) * 1e3
        for slots in sorted({max(1, int(staging_slots)), 1}, reverse=True):
            room = budget - fixed - kv_bytes - slots * layer_bytes
            if room < 0:
                continue
            stream_resident = min(num_layers - 1, int(room // layer_bytes))
            streamed = num_layers - stream_resident
            stream_ms = stream_resident * gpu_layer_ms + streamed * max(transfer_ms, gpu_layer_ms)
            if cpu_ms * _FAST_MIN_SPEEDUP > stream_ms:
                best = (stream_resident, streamed, slots)
            break

    resident, streamed, slots = best
    offloaded = list(range(resident, num_layers))
    streamed_layers = offloaded if streamed else []
    cpu_layers = [] if streamed else offloaded

    return ResidencyPlan(
        num_layers=num_layers,
        layer_bytes=layer_bytes,
        anchor_bytes=anchor_bytes,
        vram_budget_bytes=budget,
        kv_cache_reserve_bytes=int(kv_per_layer * (resident + len(streamed_layers))),
        activation_reserve_bytes=activation_reserve_bytes,
        staging_slots=slots,
        resident_layers=list(range(resident)),
        streamed_layers=streamed_layers,
        bandwidth_gb_s=bandwidth_gb_s,
        dtype_bytes=dtype_bytes,
        cpu_layers=cpu_layers,
        mode="fast",
        cpu_layer_ms=cpu_layer_ms,
        gpu_layer_ms=gpu_layer_ms,
    )


# A CPU layer's cost in real decode beyond its standalone timing: its hooks and
# moving its inputs off the GPU. Measured as the slope of decode time over the
# number of CPU layers (experiments/cpu_layer_cost.py; RTX 3060, Ryzen 5 5500):
# +0.4-0.6 ms on Qwen2.5-0.5B, -0.25 ms on 3B, +0.49 ms on 7B. It does not
# scale with the layer, so the earlier factor (1.5x) overcharged large models:
# their in-decode cost is 0.96-1.03x standalone.
_CPU_IN_DECODE_MS = 0.5

# The CPU plan gives up bit-identical output, so it has to be clearly faster.
_FAST_MIN_SPEEDUP = 1.15

# Fallbacks when a layer cannot be timed: decode is bound by memory bandwidth,
# ~26 GB/s effective from dual-channel DDR4-3200, ~100 GB/s on an RTX 3060
# with kernel-launch overhead included.
_CPU_DECODE_GB_S = 25.0
_GPU_DECODE_GB_S = 100.0


def measure_cpu_layer_ms(model: nn.Module, adapter: Optional[ModelArchAdapter] = None) -> float:
    """
    Times one decoder layer processing one token on the CPU. Falls back to a
    RAM-bandwidth estimate if the layer cannot be called standalone.
    """
    return _time_layer(model, adapter, torch.device("cpu"), _CPU_DECODE_GB_S)


def measure_gpu_layer_ms(model: nn.Module, adapter: Optional[ModelArchAdapter] = None) -> float:
    """
    Times one decoder layer processing one token on the GPU, using a temporary
    copy of a single layer. Worth the brief VRAM: for small layers kernel-launch
    overhead, not bandwidth, sets the time, and no byte count predicts that
    (0.3 ms estimated vs ~1.6 ms real on Qwen2.5-0.5B).
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return _time_layer(model, adapter, device, _GPU_DECODE_GB_S)


@torch.no_grad()
def _time_layer(
    model: nn.Module,
    adapter: Optional[ModelArchAdapter],
    device: torch.device,
    fallback_gb_s: float,
    repeats: int = 8,
) -> float:
    adapter = adapter or QwenArchitectureAdapter()
    layer = adapter.get_layers(model)[0]
    fallback = module_bytes(layer) / (fallback_gb_s * 1024**3) * 1e3
    use_cuda = device.type == "cuda"

    copied = False
    try:
        param = next(layer.parameters())
        if param.device != device:
            import copy

            layer = copy.deepcopy(layer).to(device)
            copied = True
        config = getattr(model, "config", None)
        hidden = torch.randn(1, 1, config.hidden_size, dtype=param.dtype, device=device)
        position_ids = torch.zeros(1, 1, dtype=torch.long, device=device)
        kwargs: Dict[str, Any] = {"position_ids": position_ids}
        rotary = getattr(getattr(model, "model", None), "rotary_emb", None)
        if rotary is not None:
            rotary_device = next(rotary.buffers()).device
            cos, sin = rotary(hidden.to(rotary_device), position_ids.to(rotary_device))
            kwargs["position_embeddings"] = (cos.to(device, param.dtype), sin.to(device, param.dtype))

        for _ in range(3):
            layer(hidden, **kwargs)
        if use_cuda:
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(repeats):
            layer(hidden, **kwargs)
        if use_cuda:
            torch.cuda.synchronize(device)
        return (time.perf_counter() - start) / repeats * 1e3
    except Exception:
        return fallback
    finally:
        if copied:
            del layer
            if use_cuda:
                torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# The runtime
# ---------------------------------------------------------------------------

def _move_layer_cache(cache: Any, layer_idx: int, device: torch.device) -> None:
    """Moves one layer's KV cache entries to `device` (no-op without a cache)."""
    if cache is None:
        return
    layers = getattr(cache, "layers", None)
    if layers is not None:
        if layer_idx < len(layers):
            entry = layers[layer_idx]
            for attr in ("keys", "values"):
                tensor = getattr(entry, attr, None)
                if torch.is_tensor(tensor):
                    setattr(entry, attr, tensor.to(device))
            if hasattr(entry, "device"):
                entry.device = device
        return
    # transformers < 4.56 kept parallel lists.
    for attr in ("key_cache", "value_cache"):
        tensors = getattr(cache, attr, None)
        if tensors is not None and layer_idx < len(tensors) and torch.is_tensor(tensors[layer_idx]):
            tensors[layer_idx] = tensors[layer_idx].to(device)


def _clear_cuda_error() -> None:
    """
    Consumes the CUDA runtime's pending error. torch's cudart bindings do not
    expose cudaGetLastError, so it is called from the runtime library torch
    itself loaded.
    """
    import ctypes
    import glob
    import os

    lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
    names = glob.glob(os.path.join(lib_dir, "cudart64_*.dll")) + glob.glob(
        os.path.join(lib_dir, "libcudart.so*")
    )
    names += ["libcudart.so.12", "libcudart.so"]
    for name in names:
        try:
            ctypes.CDLL(name).cudaGetLastError()
            return
        except (OSError, AttributeError):
            continue


def _to_device(value: Any, device: torch.device) -> Any:
    """Moves every tensor inside (nested) args/kwargs; leaves other objects alone."""
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_to_device(v, device) for v in value)
    if isinstance(value, list):
        return [_to_device(v, device) for v in value]
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    return value


class StreamingModel:
    """
    Installs a residency plan onto a live model.

    Resident layers and anchors are moved to the device; streamed layers stay in
    host RAM (pinned when possible, since pinned memory is what makes the DMA
    asynchronous). Each streamed layer's parameters are rebound to a rotating
    GPU staging slot immediately before that layer runs, and the transfer of the
    next layer is issued on a side stream so it overlaps the current layer's
    compute.

    Layers in `plan.cpu_layers` stay in host RAM and are computed there; hooks
    move their inputs to the CPU and their output back to the device.

    Use as a context manager; `__exit__` puts every parameter back where it was.
    """

    def __init__(
        self,
        model: nn.Module,
        plan: ResidencyPlan,
        device: Optional[torch.device] = None,
        adapter: Optional[ModelArchAdapter] = None,
        pin_memory: bool = True,
        pin_reserve: Any = "2GB",
    ):
        self.model = model
        self.plan = plan
        self.adapter = adapter or QwenArchitectureAdapter()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pin_memory = pin_memory and self.device.type == "cuda" and torch.cuda.is_available()
        # Host RAM that page-locking must leave available to the rest of the system.
        self.pin_reserve_bytes = parse_bytes(pin_reserve)
        # Flat host buffers page-locked with cudaHostRegister; unregistered on remove().
        self._registered: List[torch.Tensor] = []
        self._pin_failed = False

        # CPU layers: host tensors by layer, the GPU slot prompts run them in,
        # and which layers are borrowing that slot right now.
        self._cpu_host: Dict[int, Dict[str, torch.Tensor]] = {}
        self._prefill_slot: Optional[Dict[str, torch.Tensor]] = None
        self._on_gpu_now: set = set()
        self.prefill_layer_runs = 0

        self._layers = self.adapter.get_layers(model)
        self._installed = False
        self._handles: List[Any] = []
        # The model's class while a CPU embedding has it report the device.
        self._model_class: Optional[type] = None
        # `generation_config.prefill_chunk_size` before the plan set it.
        self._saved_chunk: Optional[Tuple[Any, Any]] = None

        # Host-side master copies of streamed parameters, by layer then name.
        self._host_params: Dict[int, Dict[str, torch.Tensor]] = {}
        self._param_refs: Dict[int, Dict[str, nn.Parameter]] = {}

        self._slots: List[Dict[str, torch.Tensor]] = []
        self._slot_ready: List[Optional[torch.cuda.Event]] = []
        self._slot_free: List[Optional[torch.cuda.Event]] = []
        self._slot_layer: List[Optional[int]] = []

        self._prefetch_stream: Optional[torch.cuda.Stream] = None
        self._stream_order: List[int] = list(plan.streamed_layers)
        self._next_to_prefetch = 0
        self._next_slot = 0
        self._assignment: Dict[int, int] = {}

        self.transfer_bytes_total = 0
        self.tokens_observed = 0

        # For a model loaded by `load_model_mmap`, every parameter's original
        # tensor is a view into the checkpoint file: holding on to it costs no
        # RAM, and restoring it on exit means the next plan starts from the
        # file again rather than from copies a previous plan left behind.
        self._originals: List[Tuple[nn.Parameter, torch.Tensor]] = []

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "StreamingModel":
        self.install()
        return self

    def __exit__(self, *exc_info) -> None:
        self.remove()

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda" and torch.cuda.is_available()

    def install(self) -> "StreamingModel":
        if self._installed:
            return self

        streamed = set(self.plan.streamed_layers)
        on_cpu = set(self.plan.cpu_layers)

        if is_file_backed(self.model):
            self._originals = [(param, param.data) for param in self.model.parameters()]

        # Anchors and resident layers go to the device. Buffers of streamed
        # layers go too: they are tiny and streaming them would buy nothing.
        # A file-backed model's pages are released as soon as their bytes have
        # been copied out (to VRAM or into pinned RAM), so that pinning the
        # next layer sees how much RAM is really available.
        # The input embedding, when the plan keeps it on the CPU, is read in
        # place (a few rows per token), so it is neither moved nor released.
        embedding = _untied_embedding(self.model) if self.plan.cpu_anchor_bytes else None
        stays_put = {id(p) for layer in self._layers for p in layer.parameters()}
        if embedding is not None:
            stays_put |= {id(p) for p in embedding.parameters()}
            embedding.to("cpu")
            self._install_embedding_hooks(embedding)
        anchor_sources = [p.data for p in self.model.parameters() if id(p) not in stays_put]
        self._move_anchors_to_device(stays_put)
        release_file_pages(self.model, anchor_sources)
        for idx, layer in enumerate(self._layers):
            sources = [p.data for p in layer.parameters()]
            if idx in streamed:
                self._move_layer_to_host(idx, layer)
                if any(t.is_pinned() for t in self._host_params[idx].values()):
                    release_file_pages(self.model, sources)
            elif idx in on_cpu:
                layer.to("cpu")
            else:
                layer.to(self.device)
                release_file_pages(self.model, sources)

        if streamed:
            self._allocate_slots()
            self._install_hooks()
        if on_cpu:
            self._install_cpu_hooks(on_cpu)

        generation_config = getattr(self.model, "generation_config", None)
        if self.plan.prefill_chunk_tokens and generation_config is not None:
            self._saved_chunk = (
                generation_config,
                getattr(generation_config, "prefill_chunk_size", None),
            )
            generation_config.prefill_chunk_size = self.plan.prefill_chunk_tokens

        self._installed = True
        return self

    def remove(self) -> None:
        if not self._installed:
            return

        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if self._model_class is not None:
            self.model.__class__ = self._model_class
            self._model_class = None
        if self._saved_chunk is not None:
            generation_config, previous = self._saved_chunk
            generation_config.prefill_chunk_size = previous
            self._saved_chunk = None

        # Rebind every streamed parameter to its host copy so the model is a
        # plain CPU/GPU model again rather than pointing into freed slots.
        for layer_idx, params in self._param_refs.items():
            host = self._host_params.get(layer_idx, {})
            for name, param in params.items():
                if name in host:
                    param.data = host[name]

        # A pass interrupted mid-prompt can leave a CPU layer bound to the slot.
        for layer_idx, host in self._cpu_host.items():
            for name, param in self._layers[layer_idx].named_parameters():
                if name in host:
                    param.data = host[name]
        self._cpu_host.clear()
        self._on_gpu_now.clear()
        self._prefill_slot = None

        for param, original in self._originals:
            param.data = original
        # Pages read during the run (layers streamed from the file, CPU
        # layers) would otherwise keep counting as used, and the next plan
        # would pin less each time (7B: 16, 13, 9, 5 layers across budgets).
        release_file_pages(self.model, [original for _, original in self._originals])
        self._originals.clear()

        # Outside a plan the host copies are ordinary RAM again: page-locked
        # memory is a system-wide cost, held only while streaming needs it.
        # In-flight copies must finish before their source is unlocked.
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
        self._unregister_host_buffers()

        self._slots.clear()
        self._slot_ready.clear()
        self._slot_free.clear()
        self._slot_layer.clear()
        self._host_params.clear()
        self._param_refs.clear()
        self._assignment.clear()
        self._prefetch_stream = None
        self._installed = False

        if self.is_cuda:
            torch.cuda.empty_cache()

    # -- placement helpers -------------------------------------------------

    def _move_anchors_to_device(self, stays_put: set) -> None:
        """Moves every parameter not in `stays_put` (ids), and the buffers outside layers."""
        layer_buffer_ids = {id(b) for layer in self._layers for b in layer.buffers()}

        for param in self.model.parameters():
            if id(param) not in stays_put and param.device != self.device:
                param.data = param.data.to(self.device)

        for module in self.model.modules():
            for key, buf in list(module._buffers.items()):
                if buf is None or id(buf) in layer_buffer_ids:
                    continue
                if buf.device != self.device:
                    module._buffers[key] = buf.to(self.device)

    def _move_layer_to_host(self, layer_idx: int, layer: nn.Module) -> None:
        host: Dict[str, torch.Tensor] = {}
        refs: Dict[str, nn.Parameter] = dict(layer.named_parameters())

        already_pinned = all(
            p.data.device.type == "cpu" and p.data.is_pinned() for p in refs.values()
        )
        if self.pin_memory and not already_pinned:
            host = self._pack_and_pin(refs)

        for name, param in refs.items():
            tensor = host.get(name)
            if tensor is None:
                tensor = param.data if param.data.device.type == "cpu" else param.data.to("cpu")
                host[name] = tensor
            param.data = tensor

        # Buffers stay on the device: they are small and needed every token.
        for module in layer.modules():
            for key, buf in list(module._buffers.items()):
                if buf is not None and buf.device != self.device:
                    module._buffers[key] = buf.to(self.device)

        self._host_params[layer_idx] = host
        self._param_refs[layer_idx] = refs

    def _pack_and_pin(self, refs: Dict[str, nn.Parameter]) -> Dict[str, torch.Tensor]:
        """
        Copies one layer's parameters into a single exactly-sized host buffer
        and page-locks it in place with cudaHostRegister.

        `Tensor.pin_memory()` goes through PyTorch's caching host allocator,
        which rounds every block up to a power of two: a 45 MB MLP matrix takes
        64 MB, about +45% across a Qwen2.5-3B layer. Registering an ordinary
        buffer pins exactly what is needed.

        Page-locked memory cannot be swapped out, so pinning stops while it
        would leave less than `pin_reserve_bytes` of RAM available; later
        layers stay pageable, which still works but transfers more slowly.
        Returns {} when the layer is left pageable.
        """
        alignment = 512
        offsets: Dict[str, int] = {}
        total = 0
        for name, param in refs.items():
            offsets[name] = total
            nbytes = param.data.numel() * param.data.element_size()
            total += -(-nbytes // alignment) * alignment

        if total == 0 or self._pin_failed or not self._can_pin(total):
            return {}

        # An anonymous mapping rather than torch.empty: torch's CPU allocator
        # keeps freed blocks for reuse (measured on Windows: 2.6 GB freed, none
        # returned), so after a plan is removed its pinned layers would still
        # count as used and the next plan could pin almost nothing. A mapping
        # is returned to the OS as soon as its last view is dropped.
        flat = torch.frombuffer(mmap.mmap(-1, total), dtype=torch.uint8)
        host: Dict[str, torch.Tensor] = {}
        for name, param in refs.items():
            src = param.data
            nbytes = src.numel() * src.element_size()
            view = flat[offsets[name] : offsets[name] + nbytes].view(src.dtype).view(src.shape)
            view.copy_(src)
            host[name] = view

        err = torch.cuda.cudart().cudaHostRegister(flat.data_ptr(), total, 0)
        if err != torch.cuda.cudart().cudaError.success:
            # The driver can refuse before RAM runs out (measured: at 7.0 GB
            # here). The failure stays pending in the CUDA runtime and the next
            # unrelated call -- allocating a staging slot -- would raise it as
            # "out of memory", so clear it, and stop trying for this plan.
            _clear_cuda_error()
            self._pin_failed = True
            # The copy is still a valid pageable host buffer; use it as such.
            return host

        self._registered.append(flat)
        return host

    def _can_pin(self, nbytes: int) -> bool:
        try:
            import psutil
        except ImportError:
            return True
        return psutil.virtual_memory().available - nbytes >= self.pin_reserve_bytes

    def _unregister_host_buffers(self) -> None:
        if not self._registered:
            return
        unregister = torch.cuda.cudart().cudaHostUnregister
        for flat in self._registered:
            unregister(flat.data_ptr())
        self._registered.clear()

    def _allocate_slots(self) -> None:
        template_idx = self.plan.streamed_layers[0]
        template = self._host_params[template_idx]

        self._prefetch_stream = torch.cuda.Stream(device=self.device) if self.is_cuda else None

        for _ in range(self.plan.staging_slots):
            slot = {
                name: torch.empty_like(tensor, device=self.device)
                for name, tensor in template.items()
            }
            self._slots.append(slot)
            self._slot_ready.append(torch.cuda.Event() if self.is_cuda else None)
            self._slot_free.append(torch.cuda.Event() if self.is_cuda else None)
            self._slot_layer.append(None)

    # -- hooks -------------------------------------------------------------

    def _install_hooks(self) -> None:
        # Fires once per forward pass, i.e. once per generated token.
        self._handles.append(self.model.register_forward_pre_hook(self._on_pass_start))

        for layer_idx in self.plan.streamed_layers:
            layer = self._layers[layer_idx]
            self._handles.append(
                layer.register_forward_pre_hook(self._make_layer_pre_hook(layer_idx))
            )
            self._handles.append(
                layer.register_forward_hook(self._make_layer_post_hook(layer_idx))
            )

    def _install_embedding_hooks(self, embedding: nn.Module) -> None:
        cpu = torch.device("cpu")
        self._handles.append(
            embedding.register_forward_pre_hook(lambda module, args: _to_device(args, cpu))
        )
        self._handles.append(
            embedding.register_forward_hook(
                lambda module, args, output: _to_device(output, self.device)
            )
        )
        # transformers reads `model.device` from the first parameter, which is
        # usually this table; `generate` would then build position ids on the
        # CPU. While the plan is installed the model reports the GPU instead.
        if isinstance(getattr(type(self.model), "device", None), property):
            self._model_class = type(self.model)
            device = self.device
            self.model.__class__ = type(
                self._model_class.__name__,
                (self._model_class,),
                {"device": property(lambda model: device)},
            )

    def _install_cpu_hooks(self, on_cpu: set) -> None:
        # Decode (one token per pass): a CPU layer receives its inputs on the
        # CPU and its output goes back to the device only at the end of a run
        # of CPU layers, so the bus is crossed twice per run, not per layer.
        # Its KV cache lives on the CPU.
        #
        # Prompts (several tokens per pass): CPU bf16 matmuls cost ~3 ms per
        # token per layer here, so if the plan kept a staging slot the layer's
        # weights are copied into it and the layer runs on the GPU. Its KV
        # cache entries are moved to the CPU afterwards, where decode expects
        # them.
        for layer_idx in sorted(on_cpu):
            self._cpu_host[layer_idx] = {
                name: param.data for name, param in self._layers[layer_idx].named_parameters()
            }

        if self.is_cuda and self.plan.staging_slots:
            template = self._cpu_host[min(on_cpu)]
            self._prefill_slot = {
                name: torch.empty_like(tensor, device=self.device) for name, tensor in template.items()
            }

        last = len(self._layers) - 1
        for layer_idx in sorted(on_cpu):
            layer = self._layers[layer_idx]
            group_end = layer_idx == last or (layer_idx + 1) not in on_cpu
            self._handles.append(
                layer.register_forward_pre_hook(self._make_cpu_pre_hook(layer_idx), with_kwargs=True)
            )
            self._handles.append(
                layer.register_forward_hook(
                    self._make_cpu_post_hook(layer_idx, group_end), with_kwargs=True
                )
            )

    def _make_cpu_pre_hook(self, layer_idx: int):
        cpu = torch.device("cpu")
        params = dict(self._layers[layer_idx].named_parameters())

        def hook(module, args, kwargs):
            hidden = args[0] if args else kwargs.get("hidden_states")
            is_prompt = torch.is_tensor(hidden) and hidden.dim() == 3 and hidden.shape[1] > 1
            if is_prompt and self._prefill_slot is not None:
                slot = self._prefill_slot
                host = self._cpu_host[layer_idx]
                # Same stream as the compute, so this waits for the previous
                # user of the slot to finish before overwriting it.
                for name, param in params.items():
                    slot[name].copy_(host[name])
                    param.data = slot[name]
                # A chunk after the first attends to what earlier chunks left
                # in this layer's cache, which the post hook moved to the CPU.
                cache_idx = getattr(getattr(module, "self_attn", None), "layer_idx", layer_idx)
                _move_layer_cache(kwargs.get("past_key_values"), cache_idx, self.device)
                self._on_gpu_now.add(layer_idx)
                self.prefill_layer_runs += 1
                self.transfer_bytes_total += self.plan.layer_bytes
                return _to_device(args, self.device), _to_device(kwargs, self.device)
            return _to_device(args, cpu), _to_device(kwargs, cpu)

        return hook

    def _make_cpu_post_hook(self, layer_idx: int, group_end: bool):
        cpu = torch.device("cpu")
        params = dict(self._layers[layer_idx].named_parameters())

        def hook(module, args, kwargs, output):
            if layer_idx in self._on_gpu_now:
                self._on_gpu_now.discard(layer_idx)
                host = self._cpu_host[layer_idx]
                for name, param in params.items():
                    param.data = host[name]
                cache_idx = getattr(getattr(module, "self_attn", None), "layer_idx", layer_idx)
                _move_layer_cache(kwargs.get("past_key_values"), cache_idx, cpu)
                return output
            return _to_device(output, self.device) if group_end else output

        return hook

    def _on_pass_start(self, module, args) -> None:
        self._next_to_prefetch = 0
        self._next_slot = 0
        self._assignment.clear()
        self.tokens_observed += 1

        # Fill the pipeline: these transfers overlap the resident layers that
        # run before the first streamed layer.
        for _ in range(len(self._slots)):
            self._prefetch_next()

    def _prefetch_next(self) -> None:
        if self._next_to_prefetch >= len(self._stream_order):
            return

        layer_idx = self._stream_order[self._next_to_prefetch]
        slot_idx = self._next_slot
        slot = self._slots[slot_idx]
        host = self._host_params[layer_idx]

        if self.is_cuda and self._prefetch_stream is not None:
            free_event = self._slot_free[slot_idx]
            # Do not overwrite a slot whose previous layer is still computing.
            if self._slot_layer[slot_idx] is not None and free_event is not None:
                self._prefetch_stream.wait_event(free_event)
            with torch.cuda.stream(self._prefetch_stream):
                for name, destination in slot.items():
                    destination.copy_(host[name], non_blocking=True)
            ready = self._slot_ready[slot_idx]
            if ready is not None:
                ready.record(self._prefetch_stream)
        else:
            for name, destination in slot.items():
                destination.copy_(host[name])

        self._slot_layer[slot_idx] = layer_idx
        self._assignment[layer_idx] = slot_idx
        self.transfer_bytes_total += self.plan.layer_bytes
        self._next_to_prefetch += 1
        self._next_slot = (self._next_slot + 1) % len(self._slots)

    def _make_layer_pre_hook(self, layer_idx: int):
        def hook(module, args):
            slot_idx = self._assignment.get(layer_idx)
            if slot_idx is None:
                # The pipeline fell behind (unexpected pass shape); load now.
                while self._next_to_prefetch < len(self._stream_order) and layer_idx not in self._assignment:
                    self._prefetch_next()
                slot_idx = self._assignment.get(layer_idx)
                if slot_idx is None:
                    raise RuntimeError(
                        f"Streamed layer {layer_idx} was reached without a staging slot."
                    )

            if self.is_cuda:
                ready = self._slot_ready[slot_idx]
                if ready is not None:
                    torch.cuda.current_stream().wait_event(ready)

            slot = self._slots[slot_idx]
            for name, param in self._param_refs[layer_idx].items():
                param.data = slot[name]
            return None

        return hook

    def _make_layer_post_hook(self, layer_idx: int):
        def hook(module, args, output):
            slot_idx = self._assignment.get(layer_idx)
            if slot_idx is not None and self.is_cuda:
                free_event = self._slot_free[slot_idx]
                if free_event is not None:
                    free_event.record(torch.cuda.current_stream())
            # Now that this slot's compute is queued, start the next transfer.
            self._prefetch_next()
            return output

        return hook

    # -- reporting ---------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "installed": self._installed,
            "resident_layers": len(self.plan.resident_layers),
            "streamed_layers": len(self.plan.streamed_layers),
            # Streamed layers whose host copy is page-locked; the rest transfer
            # synchronously from pageable RAM and are slower.
            "pinned_layers": sum(
                1
                for params in self._host_params.values()
                if params and all(t.is_pinned() for t in params.values())
            ),
            "staging_slots": self.plan.staging_slots,
            "forward_passes": self.tokens_observed,
            # CPU-layer executions moved to the GPU because the pass was a prompt.
            "prefill_layer_runs": self.prefill_layer_runs,
            "bytes_transferred": self.transfer_bytes_total,
            "vram_weight_bytes": self.plan.vram_weight_bytes,
            "host_weight_bytes": self.plan.host_weight_bytes,
        }
