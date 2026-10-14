"""
Tiered Memory Manager with Asynchronous CUDA Stream Prefetching.
Manages VRAM (Hot), Pinned Host RAM (Warm Staging), and Disk (Cold).
"""

import time
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
import torch
from ..device.profiler import HardwareBudget

@dataclass
class MemoryTierStats:
    vram_hot_bytes: int
    ram_pinned_bytes: int
    num_hot_blocks: int
    num_warm_blocks: int
    last_transfer_latency_ms: float

class TieredMemoryManager:
    """
    Hierarchical memory manager for ParamMap:
    - Tier 0: GPU VRAM (Hot)
    - Tier 1: Host Pinned Memory (Warm)
    - Tier 2: Storage Disk / mmap (Cold)
    
    Uses dedicated CUDA streams for non-blocking asynchronous DMA prefetching.
    """

    def __init__(
        self,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        hardware_budget: Optional[HardwareBudget] = None,
        max_vram_bytes: Optional[int] = None,
        max_warm_ram_bytes: Optional[int] = None
    ):
        self.device = device
        self.hardware_budget = hardware_budget

        self.max_vram_bytes = max_vram_bytes or (
            hardware_budget.recommended_hot_vram_bytes if hardware_budget else 4 * 1024**3
        )
        self.max_warm_ram_bytes = max_warm_ram_bytes or (
            hardware_budget.recommended_warm_ram_bytes if hardware_budget else 2 * 1024**3
        )

        # Streams
        self.is_cuda = (self.device.type == "cuda" and torch.cuda.is_available())
        self.prefetch_stream = torch.cuda.Stream(device=self.device) if self.is_cuda else None

        # Tier 0: (layer_idx, proj_name) -> Tensor on GPU
        # Tier 1: (layer_idx, proj_name) -> Pinned Tensor on CPU
        self.hot_tensors: Dict[Tuple[int, str], torch.Tensor] = {}
        self.warm_pinned_tensors: Dict[Tuple[int, str], torch.Tensor] = {}

        # Active resident block tracking: layer_idx -> set of block_indices
        self.hot_blocks: Dict[int, Set[int]] = {}
        self.last_transfer_ms: float = 0.0

    def register_warm_block(
        self,
        layer_idx: int,
        block_idx: int,
        gate_slice: torch.Tensor,
        up_slice: torch.Tensor,
        down_slice: torch.Tensor
    ) -> None:
        """
        Stores weight slices into Host Pinned Memory for rapid DMA transfers.
        """
        key_gate = (layer_idx, f"b{block_idx}_gate")
        key_up = (layer_idx, f"b{block_idx}_up")
        key_down = (layer_idx, f"b{block_idx}_down")

        # Pin memory on CPU
        def pin(t: torch.Tensor) -> torch.Tensor:
            t_cpu = t.detach().cpu()
            return t_cpu.pin_memory() if (self.is_cuda and not t_cpu.is_pinned()) else t_cpu

        self.warm_pinned_tensors[key_gate] = pin(gate_slice)
        self.warm_pinned_tensors[key_up] = pin(up_slice)
        self.warm_pinned_tensors[key_down] = pin(down_slice)

    def prefetch_to_hot_async(
        self,
        layer_idx: int,
        active_block_indices: torch.Tensor,
        full_weights_cpu: Dict[str, torch.Tensor],
        block_size: int,
        neuron_order: torch.Tensor
    ) -> None:
        """
        Slices requested blocks and transfers them to GPU memory asynchronously
        using the secondary CUDA prefetch stream without blocking compute.
        """
        t0 = time.perf_counter()

        # Build contiguous active indices
        active_indices_list = []
        for b_idx in active_block_indices.tolist():
            s = b_idx * block_size
            e = s + block_size
            active_indices_list.append(neuron_order[s:e])

        if not active_indices_list:
            self.hot_blocks[layer_idx] = set()
            self.last_transfer_ms = (time.perf_counter() - t0) * 1000.0
            return

        active_indices = torch.cat(active_indices_list)

        # Slice on CPU (using pinned memory if available)
        gate_w = full_weights_cpu["gate"]
        up_w = full_weights_cpu["up"]
        down_w = full_weights_cpu["down"]

        gate_slice = gate_w.index_select(0, active_indices)
        up_slice = up_w.index_select(0, active_indices)
        if down_w.shape[1] == len(neuron_order):
            down_slice = down_w.index_select(1, active_indices)
        else:
            down_slice = down_w.index_select(0, active_indices)

        # Transfer asynchronously on prefetch stream
        if self.is_cuda and self.prefetch_stream:
            with torch.cuda.stream(self.prefetch_stream):
                gate_gpu = gate_slice.to(self.device, non_blocking=True)
                up_gpu = up_slice.to(self.device, non_blocking=True)
                down_gpu = down_slice.to(self.device, non_blocking=True)
        else:
            gate_gpu = gate_slice.to(self.device)
            up_gpu = up_slice.to(self.device)
            down_gpu = down_slice.to(self.device)

        self.hot_tensors[(layer_idx, "gate")] = gate_gpu
        self.hot_tensors[(layer_idx, "up")] = up_gpu
        self.hot_tensors[(layer_idx, "down")] = down_gpu
        self.hot_blocks[layer_idx] = set(active_block_indices.tolist())

        self.last_transfer_ms = (time.perf_counter() - t0) * 1000.0

    def synchronize(self) -> None:
        """
        Synchronizes the current default compute stream with the prefetch stream,
        guaranteeing all DMA transfers are complete before computation begins.
        """
        if self.is_cuda and self.prefetch_stream:
            torch.cuda.current_stream().wait_stream(self.prefetch_stream)

    def get_layer_hot_weights(self, layer_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Returns the active resident GPU tensors for layer_idx.
        """
        gate = self.hot_tensors.get((layer_idx, "gate"))
        up = self.hot_tensors.get((layer_idx, "up"))
        down = self.hot_tensors.get((layer_idx, "down"))

        if gate is not None and up is not None and down is not None:
            return {"gate": gate, "up": up, "down": down}
        return None

    def evict_layer(self, layer_idx: int) -> None:
        """
        Evicts layer's hot tensors from VRAM.
        """
        for proj in ["gate", "up", "down"]:
            key = (layer_idx, proj)
            if key in self.hot_tensors:
                del self.hot_tensors[key]
        self.hot_blocks[layer_idx] = set()

    def get_stats(self) -> MemoryTierStats:
        """
        Calculates memory footprints across VRAM and Pinned RAM.
        """
        vram_bytes = sum(t.element_size() * t.numel() for t in self.hot_tensors.values())
        pinned_bytes = sum(t.element_size() * t.numel() for t in self.warm_pinned_tensors.values())
        num_hot = sum(len(b) for b in self.hot_blocks.values())
        num_warm = len(self.warm_pinned_tensors) // 3

        return MemoryTierStats(
            vram_hot_bytes=vram_bytes,
            ram_pinned_bytes=pinned_bytes,
            num_hot_blocks=num_hot,
            num_warm_blocks=num_warm,
            last_transfer_latency_ms=round(self.last_transfer_ms, 2)
        )
