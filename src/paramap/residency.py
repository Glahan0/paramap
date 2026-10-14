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
"""

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .arch.base import ModelArchAdapter
from .arch.qwen import QwenArchitectureAdapter

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


def estimate_activation_reserve_bytes(
    model: nn.Module,
    config: Any,
    dtype_bytes: int = 2,
) -> int:
    """
    Headroom to leave for the transient tensors of a forward pass.

    A heuristic, deliberately a modest one: during decode the per-token
    activations are negligible and the only sizeable transient is the output
    logits row (vocabulary x dtype). The term that actually scales is prefill
    scratch, which tracks model size, so this allows a small fraction of the
    weights plus room for a batch of logits rows.

    It is a starting point, not a measurement. `paramap.benchmark` reports the
    real peak VRAM, so tune `activation_reserve` from that if a workload needs
    more or can live with less.
    """
    total = module_bytes(model)
    vocab_size = getattr(config, "vocab_size", 0) if config is not None else 0
    logits_headroom = vocab_size * dtype_bytes * 16
    return max(64 * 1024**2, int(0.02 * total) + logits_headroom)


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
    never streamed.
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

    @property
    def total_weight_bytes(self) -> int:
        return self.anchor_bytes + self.num_layers * self.layer_bytes

    @property
    def staging_bytes(self) -> int:
        return self.staging_slots * self.layer_bytes if self.streamed_layers else 0

    @property
    def resident_weight_bytes(self) -> int:
        return self.anchor_bytes + len(self.resident_layers) * self.layer_bytes

    @property
    def vram_weight_bytes(self) -> int:
        """Weight bytes actually held in VRAM, staging buffers included."""
        return self.resident_weight_bytes + self.staging_bytes

    @property
    def host_weight_bytes(self) -> int:
        """Weight bytes kept in host RAM."""
        return len(self.streamed_layers) * self.layer_bytes

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
    def is_fully_resident(self) -> bool:
        return not self.streamed_layers

    def describe(self) -> str:
        lines = [
            f"VRAM budget            : {format_bytes(self.vram_budget_bytes)}",
            f"Model weights (total)  : {format_bytes(self.total_weight_bytes)}"
            f"  ({self.num_layers} layers x {format_bytes(self.layer_bytes)}"
            f" + {format_bytes(self.anchor_bytes)} anchors)",
            "",
            f"  resident layers      : {len(self.resident_layers)}/{self.num_layers}"
            f"  ({format_bytes(len(self.resident_layers) * self.layer_bytes)})",
            f"  streamed layers      : {len(self.streamed_layers)}/{self.num_layers}"
            f"  ({format_bytes(self.host_weight_bytes)} held in host RAM)",
            f"  anchors (always VRAM): {format_bytes(self.anchor_bytes)}",
            f"  staging slots        : {self.staging_slots}"
            f"  ({format_bytes(self.staging_bytes)})",
            "",
            f"  weights in VRAM      : {format_bytes(self.vram_weight_bytes)}",
            f"  KV cache reserved    : {format_bytes(self.kv_cache_reserve_bytes)}",
            f"  activation reserve   : {format_bytes(self.activation_reserve_bytes)}",
            f"  VRAM saved vs all-in : {format_bytes(self.vram_saved_bytes)}"
            f"  ({self.vram_saved_percent:.1f}%)",
        ]
        if self.is_fully_resident:
            lines += ["", "  Nothing needs streaming at this budget: full speed."]
        else:
            lines += [
                "",
                f"  Streamed per token   : {format_bytes(self.bytes_streamed_per_token)}",
                f"  Bus speed            : {self.bandwidth_gb_s:.2f} GB/s",
                f"  Speed ceiling        : {self.tokens_per_s_ceiling:.1f} tokens/s"
                " (bus alone, best case)",
            ]
        return "\n".join(lines)


class InsufficientVRAMError(RuntimeError):
    """Raised when a budget cannot even hold the always-resident parts."""


def plan_residency(
    model: nn.Module,
    vram_budget: Any,
    *,
    max_context_tokens: int = 4096,
    staging_slots: int = 2,
    bandwidth_gb_s: Optional[float] = None,
    activation_reserve: Any = "auto",
    adapter: Optional[ModelArchAdapter] = None,
) -> ResidencyPlan:
    """
    Decides how many decoder layers fit in `vram_budget` and how many must be
    streamed from host RAM.

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
    anchor_bytes = max(0, total_model_bytes - sum(layer_sizes))

    first_param = next(model.parameters(), None)
    dtype_bytes = first_param.element_size() if first_param is not None else 2

    config = getattr(model, "config", None)
    kv_bytes = (
        estimate_kv_cache_bytes(config, max_context_tokens, dtype_bytes) if config is not None else 0
    )
    activation_reserve_bytes = (
        estimate_activation_reserve_bytes(model, config, dtype_bytes)
        if str(activation_reserve) == "auto"
        else parse_bytes(activation_reserve)
    )

    if bandwidth_gb_s is None:
        from .device.profiler import DeviceProfiler

        bandwidth_gb_s = DeviceProfiler().benchmark_bandwidth().get("h2d_gb_s", 0.0)

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


# ---------------------------------------------------------------------------
# The runtime
# ---------------------------------------------------------------------------

class StreamingModel:
    """
    Installs a residency plan onto a live model.

    Resident layers and anchors are moved to the device; streamed layers stay in
    host RAM (pinned when possible, since pinned memory is what makes the DMA
    asynchronous). Each streamed layer's parameters are rebound to a rotating
    GPU staging slot immediately before that layer runs, and the transfer of the
    next layer is issued on a side stream so it overlaps the current layer's
    compute.

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

        self._layers = self.adapter.get_layers(model)
        self._installed = False
        self._handles: List[Any] = []

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

        # Anchors and resident layers go to the device. Buffers of streamed
        # layers go too: they are tiny and streaming them would buy nothing.
        self._move_anchors_to_device()
        for idx, layer in enumerate(self._layers):
            if idx in streamed:
                self._move_layer_to_host(idx, layer)
            else:
                layer.to(self.device)

        if streamed:
            self._allocate_slots()
            self._install_hooks()

        self._installed = True
        return self

    def remove(self) -> None:
        if not self._installed:
            return

        for handle in self._handles:
            handle.remove()
        self._handles.clear()

        # Rebind every streamed parameter to its host copy so the model is a
        # plain CPU/GPU model again rather than pointing into freed slots.
        for layer_idx, params in self._param_refs.items():
            host = self._host_params.get(layer_idx, {})
            for name, param in params.items():
                if name in host:
                    param.data = host[name]

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

    def _move_anchors_to_device(self) -> None:
        layer_param_ids = {id(p) for layer in self._layers for p in layer.parameters()}
        layer_buffer_ids = {id(b) for layer in self._layers for b in layer.buffers()}

        for param in self.model.parameters():
            if id(param) not in layer_param_ids and param.device != self.device:
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

        if total == 0 or not self._can_pin(total):
            return {}

        flat = torch.empty(total, dtype=torch.uint8)
        host: Dict[str, torch.Tensor] = {}
        for name, param in refs.items():
            src = param.data
            nbytes = src.numel() * src.element_size()
            view = flat[offsets[name] : offsets[name] + nbytes].view(src.dtype).view(src.shape)
            view.copy_(src)
            host[name] = view

        err = torch.cuda.cudart().cudaHostRegister(flat.data_ptr(), total, 0)
        if err != torch.cuda.cudart().cudaError.success:
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
            "bytes_transferred": self.transfer_bytes_total,
            "vram_weight_bytes": self.plan.vram_weight_bytes,
            "host_weight_bytes": self.plan.host_weight_bytes,
        }
