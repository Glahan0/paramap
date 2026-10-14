"""
Multi-Objective Block Scheduler for ParamMap.
Calculates active block configurations balancing quality, memory footprint,
and PCIe transfer overhead under hysteresis control.
"""

import warnings
from enum import Enum
from typing import Dict, List, Set, Tuple, Optional
import torch
from dataclasses import dataclass
from .mapper import ParameterMap
from ..device.profiler import HardwareBudget

class SchedulingProfile(Enum):
    QUALITY = "quality"         # ~75% - 90% blocks active
    BALANCED = "balanced"       # ~40% - 55% blocks active
    SPEED = "speed"             # ~25% - 35% blocks active
    MIN_MEMORY = "min_memory"   # ~15% - 25% blocks active

PROFILE_BUDGET_RATIOS = {
    SchedulingProfile.QUALITY: 0.80,
    SchedulingProfile.BALANCED: 0.50,
    SchedulingProfile.SPEED: 0.30,
    SchedulingProfile.MIN_MEMORY: 0.20,
}

@dataclass
class ScheduleDecision:
    hot_blocks_per_layer: List[torch.Tensor]    # Active block indices on GPU
    evicted_blocks_per_layer: List[torch.Tensor]
    loaded_blocks_per_layer: List[torch.Tensor]
    active_ratio: float
    estimated_vram_bytes: int

class BlockScheduler:
    """
    Decides which parameter blocks are resident in VRAM for each layer.
    Incorporates hysteresis to prevent oscillatory loading/eviction across turns.
    """

    def __init__(
        self,
        parameter_map: ParameterMap,
        hardware_budget: Optional[HardwareBudget] = None,
        profile: SchedulingProfile = SchedulingProfile.BALANCED,
        hysteresis_decay: float = 0.85
    ):
        self.param_map = parameter_map
        self.hardware_budget = hardware_budget
        self.profile = profile
        self.hysteresis_decay = hysteresis_decay

        # State tracking: layer_idx -> tensor of block retention weights
        self.resident_hot_blocks: List[Set[int]] = [
            set() for _ in range(self.param_map.num_layers)
        ]
        self._warned_no_domain_overlap = False

    def set_profile(self, profile: SchedulingProfile) -> None:
        self.profile = profile

    def schedule(
        self,
        domain_scores: Dict[str, float],
        override_budget_ratio: Optional[float] = None
    ) -> ScheduleDecision:
        """
        Computes the target set of hot blocks per layer based on domain affinity.
        """
        target_ratio = override_budget_ratio or PROFILE_BUDGET_RATIOS[self.profile]
        num_blocks = self.param_map.num_blocks
        k_blocks = max(1, int(num_blocks * target_ratio))

        # Build domain weight vector matching param_map.metadata.domain_ids
        domain_ids = self.param_map.metadata.domain_ids
        domain_weights = torch.tensor(
            [domain_scores.get(d_id, 0.0) for d_id in domain_ids],
            dtype=torch.float32
        )
        total_weight = float(domain_weights.sum())
        if total_weight > 0:
            domain_weights = domain_weights / total_weight
        else:
            # No overlap between the caller's domain ids and the map's. The
            # uniform fallback below makes block selection context-independent,
            # so say so instead of degrading silently.
            if not self._warned_no_domain_overlap:
                warnings.warn(
                    "None of the supplied domain scores match this parameter map's "
                    f"domain ids {list(domain_ids)}; block selection falls back to a "
                    "uniform, context-independent schedule.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_no_domain_overlap = True
            domain_weights = torch.full((len(domain_ids),), 1.0 / len(domain_ids))

        # Composite score: [L, NumBlocks] = sum_d (block_scores[L, B, d] * w[d])
        # block_scores: [L, NumBlocks, NumDomains]
        composite_scores = torch.einsum("lbd,d->lb", self.param_map.block_scores, domain_weights)

        hot_blocks_list: List[torch.Tensor] = []
        evicted_list: List[torch.Tensor] = []
        loaded_list: List[torch.Tensor] = []

        total_active_blocks = 0

        for l in range(self.param_map.num_layers):
            layer_scores = composite_scores[l].clone()

            # Apply hysteresis bonus to currently resident blocks
            for block_idx in self.resident_hot_blocks[l]:
                layer_scores[block_idx] += self.hysteresis_decay * 0.15

            # Always reserve top 10% core blocks regardless of domain
            core_k = max(1, int(num_blocks * 0.10))
            core_blocks = torch.topk(self.param_map.block_scores[l].mean(dim=-1), k=core_k).indices

            # Select top-k domain composite blocks
            domain_top_k = torch.topk(layer_scores, k=k_blocks).indices

            # Union
            new_hot_set = set(torch.unique(torch.cat([core_blocks, domain_top_k])).tolist())
            prev_hot_set = self.resident_hot_blocks[l]

            loaded = new_hot_set - prev_hot_set
            evicted = prev_hot_set - new_hot_set

            self.resident_hot_blocks[l] = new_hot_set

            hot_tensor = torch.tensor(sorted(list(new_hot_set)), dtype=torch.long)
            hot_blocks_list.append(hot_tensor)
            evicted_list.append(torch.tensor(sorted(list(evicted)), dtype=torch.long))
            loaded_list.append(torch.tensor(sorted(list(loaded)), dtype=torch.long))

            total_active_blocks += len(new_hot_set)

        total_possible = self.param_map.num_layers * num_blocks
        active_ratio = total_active_blocks / max(1, total_possible)

        # Estimate memory (gate + up + down projections for active blocks in fp16/bf16 = 2 bytes)
        # Each block has: block_size * hidden_size * 3 weights * 2 bytes
        intermediate_size = self.param_map.metadata.intermediate_size
        block_size = self.param_map.block_size
        # Approximate hidden_size ~ intermediate_size / 2.7
        approx_hidden = int(intermediate_size / 2.7)
        bytes_per_block = block_size * approx_hidden * 3 * 2
        estimated_vram = total_active_blocks * bytes_per_block

        return ScheduleDecision(
            hot_blocks_per_layer=hot_blocks_list,
            evicted_blocks_per_layer=evicted_list,
            loaded_blocks_per_layer=loaded_list,
            active_ratio=active_ratio,
            estimated_vram_bytes=estimated_vram,
        )
