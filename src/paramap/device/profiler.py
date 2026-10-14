"""
Hardware and Device Profiler for ParamMap.
Detects available compute devices, memory limits, and profiles transfer bandwidth.
"""

import time
import os
import psutil
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class HardwareBudget:
    device_type: str            # 'cuda', 'mps', 'cpu'
    device_name: str
    total_ram_bytes: int
    free_ram_bytes: int
    total_vram_bytes: int
    free_vram_bytes: int
    h2d_bandwidth_gb_s: float   # Host-to-Device transfer speed
    d2h_bandwidth_gb_s: float   # Device-to-Host transfer speed
    recommended_hot_vram_bytes: int
    recommended_warm_ram_bytes: int

class DeviceProfiler:
    def __init__(self):
        self.device_type = self._detect_device_type()
        
    def _detect_device_type(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def profile_memory(self) -> Dict[str, Any]:
        """Gathers system RAM and GPU VRAM statistics."""
        vm = psutil.virtual_memory()
        mem_info = {
            "ram_total_bytes": vm.total,
            "ram_free_bytes": vm.available,
            "vram_total_bytes": 0,
            "vram_free_bytes": 0,
            "gpu_name": "None",
            "device_type": self.device_type,
        }

        if self.device_type == "cuda":
            try:
                import torch
                dev_idx = torch.cuda.current_device()
                mem_info["gpu_name"] = torch.cuda.get_device_name(dev_idx)
                free_b, total_b = torch.cuda.mem_get_info(dev_idx)
                mem_info["vram_free_bytes"] = free_b
                mem_info["vram_total_bytes"] = total_b
            except Exception as e:
                mem_info["gpu_error"] = str(e)

        return mem_info

    def benchmark_bandwidth(self, test_size_mb: int = 128) -> Dict[str, float]:
        """
        Benchmarks Host-to-Device (PCIe / Bus) and Device-to-Host bandwidth in GB/s.
        Uses pinned memory for accurate PCIe throughput measurement.
        """
        results = {"h2d_gb_s": 0.0, "d2h_gb_s": 0.0}
        if self.device_type != "cuda":
            return results

        try:
            import torch
            num_bytes = test_size_mb * 1024 * 1024
            num_elements = num_bytes // 4

            # Pinned host tensor
            host_pinned = torch.empty(num_elements, dtype=torch.float32, pin_memory=True)
            device_tensor = torch.empty(num_elements, dtype=torch.float32, device="cuda")

            # Warmup
            device_tensor.copy_(host_pinned, non_blocking=True)
            torch.cuda.synchronize()

            # Host to Device
            start = time.perf_counter()
            for _ in range(5):
                device_tensor.copy_(host_pinned, non_blocking=True)
                torch.cuda.synchronize()
            elapsed_h2d = (time.perf_counter() - start) / 5.0
            results["h2d_gb_s"] = (num_bytes / (1024**3)) / elapsed_h2d

            # Device to Host
            start = time.perf_counter()
            for _ in range(5):
                host_pinned.copy_(device_tensor, non_blocking=True)
                torch.cuda.synchronize()
            elapsed_d2h = (time.perf_counter() - start) / 5.0
            results["d2h_gb_s"] = (num_bytes / (1024**3)) / elapsed_d2h

            del host_pinned, device_tensor
            torch.cuda.empty_cache()
        except Exception:
            pass

        return results

    def get_hardware_budget(self, reserve_vram_ratio: float = 0.25) -> HardwareBudget:
        """
        Computes conservative budget for parameter allocation.
        `reserve_vram_ratio` keeps headroom for KV cache and activations.
        """
        mem = self.profile_memory()
        bandwidth = self.benchmark_bandwidth()

        free_vram = mem["vram_free_bytes"]
        # Leave reserve for KV cache and activation scratchpad
        rec_hot_vram = int(free_vram * (1.0 - reserve_vram_ratio)) if free_vram > 0 else 0

        # In system RAM, leave at least 2 GB for the OS and general processes
        free_ram = mem["ram_free_bytes"]
        rec_warm_ram = max(0, free_ram - (2 * 1024 * 1024 * 1024))

        return HardwareBudget(
            device_type=self.device_type,
            device_name=mem["gpu_name"],
            total_ram_bytes=mem["ram_total_bytes"],
            free_ram_bytes=mem["ram_free_bytes"],
            total_vram_bytes=mem["vram_total_bytes"],
            free_vram_bytes=mem["vram_free_bytes"],
            h2d_bandwidth_gb_s=round(bandwidth["h2d_gb_s"], 2),
            d2h_bandwidth_gb_s=round(bandwidth["d2h_gb_s"], 2),
            recommended_hot_vram_bytes=rec_hot_vram,
            recommended_warm_ram_bytes=rec_warm_ram,
        )
