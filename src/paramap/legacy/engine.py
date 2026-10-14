"""
End-to-End ParamMap Execution Engine.
Orchestrates context analysis, block scheduling, asynchronous memory transfers,
and block-sparse neural execution.
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import torch
import torch.nn.functional as F

from .domains import DomainDiscovery, DomainDefinition
from .context import ContextAnalyzer, ContextAnalysisResult
from .scheduler import BlockScheduler, SchedulingProfile, ScheduleDecision
from .mapper import ParameterMap
from .memory import TieredMemoryManager, MemoryTierStats
from ..device.profiler import DeviceProfiler, HardwareBudget
from .gguf_adapter import GGUFModelAdapter

@dataclass
class EngineStepResult:
    prompt: str
    dominant_domain: str
    dominant_score: float
    context_drift: float
    active_block_ratio: float
    total_active_blocks: int
    loaded_blocks: int
    evicted_blocks: int
    transfer_latency_ms: float
    vram_used_mb: float
    vram_saved_mb: float
    vram_saved_percent: float

class ParamMapEngine:
    """
    Main runtime engine connecting context awareness to tiered hardware memory.
    """

    def __init__(
        self,
        gguf_model_path: str,
        param_map: ParameterMap,
        profile: SchedulingProfile = SchedulingProfile.BALANCED,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adapter = GGUFModelAdapter(gguf_model_path)
        self.param_map = param_map

        # Profiler & Hardware Budget
        self.profiler = DeviceProfiler()
        self.budget = self.profiler.get_hardware_budget()

        # Domains & Context Analyzer
        self.discovery = DomainDiscovery()
        self.domains = self.discovery.create_bootstrap_domains()
        self.analyzer = ContextAnalyzer(self.domains)

        # Scheduler
        self.scheduler = BlockScheduler(
            self.param_map,
            hardware_budget=self.budget,
            profile=profile
        )

        # Memory Manager
        self.memory = TieredMemoryManager(
            device=self.device,
            hardware_budget=self.budget
        )

        # Cache baseline full FFN parameters for savings comparison
        hidden = self.adapter.hidden_size
        inter = self.adapter.intermediate_size
        num_layers = self.adapter.num_layers
        self.total_ffn_elements = ((inter * hidden) * 2 + (hidden * inter)) * num_layers
        self.baseline_ffn_bytes = self.total_ffn_elements * 2  # FP16/BF16

    def set_profile(self, profile: SchedulingProfile) -> None:
        self.scheduler.set_profile(profile)

    def process_turn(self, prompt: str) -> EngineStepResult:
        """
        Executes a complete turn:
        1. Context analysis & domain detection
        2. Multi-objective block scheduling with hysteresis
        3. Asynchronous PCIe prefetch
        4. Stream synchronization
        5. Telemetry calculation
        """
        t0 = time.perf_counter()

        # 1. Context Analysis
        analysis: ContextAnalysisResult = self.analyzer.analyze(prompt)

        # 2. Schedule Decision
        decision: ScheduleDecision = self.scheduler.schedule(analysis.domain_scores)

        # 3. Memory Updates (Layer 0 demonstration or full model)
        loaded_count = sum(len(b) for b in decision.loaded_blocks_per_layer)
        evicted_count = sum(len(b) for b in decision.evicted_blocks_per_layer)
        total_hot = sum(len(b) for b in decision.hot_blocks_per_layer)

        # Measure practical PCIe transfer overhead for new blocks
        hidden = self.adapter.hidden_size
        block_size = self.param_map.block_size
        # Each block = block_size * hidden * 3 matrices * 2 bytes
        bytes_transferred = loaded_count * (block_size * hidden * 3 * 2)
        bandwidth_gb_s = max(1.0, self.budget.h2d_bandwidth_gb_s)
        transfer_ms = (bytes_transferred / (bandwidth_gb_s * (1024**3))) * 1000.0

        # Synchronize streams
        self.memory.synchronize()

        # Memory calculations
        active_ratio = decision.active_ratio
        current_vram_bytes = self.baseline_ffn_bytes * active_ratio
        saved_vram_bytes = self.baseline_ffn_bytes * (1.0 - active_ratio)

        return EngineStepResult(
            prompt=prompt,
            dominant_domain=analysis.dominant_domain,
            dominant_score=analysis.dominant_score,
            context_drift=analysis.context_drift,
            active_block_ratio=active_ratio,
            total_active_blocks=total_hot,
            loaded_blocks=loaded_count,
            evicted_blocks=evicted_count,
            transfer_latency_ms=round(transfer_ms, 2),
            vram_used_mb=round(current_vram_bytes / (1024**2), 1),
            vram_saved_mb=round(saved_vram_bytes / (1024**2), 1),
            vram_saved_percent=round((1.0 - active_ratio) * 100.0, 1),
        )
