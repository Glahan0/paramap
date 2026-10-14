"""
ParamMap On-Disk GGUF Experiment (Qwen3.5-9B & Gemma-4-12B).
Demonstrates block slicing, dynamic memory budgeting, and PCIe transfer latency
directly using the models on the local disk.
"""

import os
import time
import torch
from paramap.arch import GGUFModelAdapter
from paramap.device import DeviceProfiler
from paramap.domains import DomainDiscovery
from paramap.context import ContextAnalyzer
from paramap.mapper import MapBuilder, ParameterMap
from paramap.scheduler import BlockScheduler, SchedulingProfile

def main():
    qwen_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf")
    gemma_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/gemma-4-12B-it-QAT-GGUF/gemma-4-12B-it-QAT-Q4_0.gguf")

    print("=" * 75)
    print("      PARAMAP: ON-DISK GGUF MODELS ARCHITECTURE & SCHEDULING")
    print("=" * 75)

    profiler = DeviceProfiler()
    budget = profiler.get_hardware_budget()
    print(f"Device: {budget.device_name} ({budget.device_type.upper()})")
    print(f"Current Free VRAM: {budget.free_vram_bytes / (1024**3):.2f} GB / {budget.total_vram_bytes / (1024**3):.2f} GB")
    print(f"PCIe Transfer Bandwidth: {budget.h2d_bandwidth_gb_s:.2f} GB/s")

    # 1. Summary of on-disk models
    print("\n[1/4] Inspecting On-Disk Models:")
    for path, label in [(qwen_path, "Qwen3.5-9B"), (gemma_path, "Gemma-4-12B")]:
        if os.path.exists(path):
            adapter = GGUFModelAdapter(path)
            s = adapter.get_summary()
            print(f"  * {label}:")
            print(f"    - File Size: {s['file_size_gb']} GB")
            print(f"    - Architecture: {s['architecture']}")
            print(f"    - Layers: {s['num_layers']} | Hidden: {s['hidden_size']} | Intermediate: {s['intermediate_size']}")
            num_blocks_64 = s["intermediate_size"] // 64
            print(f"    - Functional Blocks per layer (size=64): {num_blocks_64} blocks (Total: {num_blocks_64 * s['num_layers']} blocks)")

    # 2. Deep Dive on Qwen3.5-9B
    print("\n[2/4] Analyzing Parameter Footprint for Qwen3.5-9B:")
    qwen_adapter = GGUFModelAdapter(qwen_path)
    
    # Calculate exact unquantized & quantized FFN weights
    hidden = qwen_adapter.hidden_size
    inter = qwen_adapter.intermediate_size
    num_layers = qwen_adapter.num_layers
    
    # Each layer has: gate (inter x hidden) + up (inter x hidden) + down (hidden x inter)
    ffn_elements_per_layer = (inter * hidden) * 2 + (hidden * inter)
    total_ffn_elements = ffn_elements_per_layer * num_layers
    fp16_ffn_bytes = total_ffn_elements * 2  # 2 bytes for fp16/bf16
    
    print(f"  - Total FFN Parameters: {total_ffn_elements / 1e9:.2f} Billion weights")
  
    print(f"  - Baseline FFN VRAM (FP16/BF16): {fp16_ffn_bytes / (1024**3):.2f} GB")

    print("\n  ParamMap Dynamic Memory Profiles (FFN Resident VRAM):")
    profiles = [
        (SchedulingProfile.QUALITY, "Quality (80% blocks)", 0.80),
        (SchedulingProfile.BALANCED, "Balanced (50% blocks)", 0.50),
        (SchedulingProfile.SPEED, "Speed (30% blocks)", 0.30),
        (SchedulingProfile.MIN_MEMORY, "Min-Memory (20% blocks)", 0.20),
    ]

    for prof, label, ratio in profiles:
        active_vram = fp16_ffn_bytes * ratio
        saved_vram = fp16_ffn_bytes * (1.0 - ratio)
        print(f"    * {label:<24}: {active_vram / (1024**3):.2f} GB VRAM (Saves {saved_vram / (1024**3):.2f} GB / {((1-ratio)*100):.0f}%)")

    # 3. Micro-benchmark: On-Demand Slicing & Transfer Speed
    print("\n[3/4] Benchmarking Block Slicing & GPU Loading (Layer 0)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_blocks = inter // 64
    active_half = torch.arange(0, num_blocks // 2, dtype=torch.long)

    t0 = time.perf_counter()
    sliced = qwen_adapter.slice_mlp_blocks(
        layer_idx=0,
        active_block_indices=active_half,
        block_size=64,
        device=device,
        dtype=torch.float16
    )
    t_slice = (time.perf_counter() - t0) * 1000.0

    print(f"  - Sliced {sliced['num_active_neurons']} neurons ({len(active_half)} blocks) for Layer 0")
    print(f"  - Slicing, Dequantizing & GPU Upload Latency: {t_slice:.2f} ms")
    print(f"  - Resident Gate Tensor on {device}: {sliced['gate'].shape} ({sliced['gate'].element_size() * sliced['gate'].numel() / (1024**2):.2f} MB)")

    # 4. Multi-Turn Turn-by-Turn Dynamic Scheduling Simulation
    print("\n[4/4] Multi-Turn Conversational Simulation on Qwen3.5-9B:")
    discovery = DomainDiscovery()
    domains = discovery.create_bootstrap_domains()
    analyzer = ContextAnalyzer(domains)

    # Synthetic block map topology for Qwen3.5-9B (32 layers, 192 blocks/layer)
    num_blocks_qwen = inter // 64
    builder = MapBuilder(block_size=64)
    # Generate structured domain selectivity
    domain_acts = {
        "math_formal": torch.randn(num_layers, inter).abs(),
        "code_python_system": torch.randn(num_layers, inter).abs(),
        "general_conversation": torch.randn(num_layers, inter).abs(),
        "creative_writing": torch.randn(num_layers, inter).abs(),
    }
    # Add domain bias
    domain_acts["math_formal"][:, :inter//3] *= 3.0
    domain_acts["code_python_system"][:, inter//3:2*inter//3] *= 3.0
    domain_acts["general_conversation"][:, 2*inter//3:] *= 3.0

    qwen_map = builder.build_from_activations(domain_acts, model_name="Qwen3.5-9B-GGUF")
    scheduler = BlockScheduler(qwen_map, hardware_budget=budget, profile=SchedulingProfile.BALANCED)

    turns = [
        ("Turn 1 (Math)", "Calculate the eigenvalue decomposition of a 4x4 matrix and its characteristic polynomial."),
        ("Turn 2 (Math Türkçe)", "Ayrıca bu matrisin determinantını ve tersini de adım adım bulun."),
        ("Turn 3 (Code Shift)", "Now write a Python multiprocessing worker pool using shared memory."),
        ("Turn 4 (General Shift)", "Great job! What are some relaxing books you recommend for the weekend?"),
    ]

    for label, prompt in turns:
        print("\n" + "-" * 70)
        print(f"{label}: \"{prompt}\"")
        analysis = analyzer.analyze(prompt)
        decision = scheduler.schedule(analysis.domain_scores)

        loaded_blocks = sum(len(b) for b in decision.loaded_blocks_per_layer)
        evicted_blocks = sum(len(b) for b in decision.evicted_blocks_per_layer)
        # Approximate transfer time via measured PCIe bandwidth
        bytes_transferred = loaded_blocks * (64 * hidden * 3 * 2)
        transfer_ms = (bytes_transferred / (budget.h2d_bandwidth_gb_s * (1024**3))) * 1000.0 if budget.h2d_bandwidth_gb_s > 0 else 0.0

        print(f"  -> Dominant Domain: {analysis.dominant_domain} ({analysis.dominant_score:.1%})")
        print(f"  -> Active Blocks: {sum(len(b) for b in decision.hot_blocks_per_layer)} / {num_layers * num_blocks_qwen} ({decision.active_ratio:.0%})")
        print(f"  -> Delta: +{loaded_blocks} blocks loaded, -{evicted_blocks} blocks evicted")
        print(f"  -> PCIe Transfer Overhead: {transfer_ms:.2f} ms")
        saved_gb = (fp16_ffn_bytes * (1.0 - decision.active_ratio)) / (1024**3)
        print(f"  -> Net VRAM Saved for Context/KV-Cache: {saved_gb:.2f} GB ({((1.0 - decision.active_ratio) * 100):.0f}%)")

    print("\n" + "=" * 75)
    print("On-Disk GGUF Models Experiment Completed Successfully!")

if __name__ == "__main__":
    main()
