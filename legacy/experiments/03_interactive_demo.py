"""
Interactive End-to-End Simulation Demo for ParamMap.
Demonstrates dynamic turn-by-turn domain analysis, scheduler decisions,
and memory savings across conversational topic shifts.
"""

import sys
from paramap.domains import DomainDiscovery
from paramap.context import ContextAnalyzer
from paramap.mapper import ParameterMap
from paramap.scheduler import BlockScheduler, SchedulingProfile
from paramap.device import DeviceProfiler

def main():
    map_path = "maps/qwen2.5_0.5b_param_map.safetensors"
    print("=" * 70)
    print("       PARAMAP: DYNAMIC ACTIVATION SCHEDULER DEMO")
    print("=" * 70)

    profiler = DeviceProfiler()
    budget = profiler.get_hardware_budget()
    print(f"Device: {budget.device_name} ({budget.device_type.upper()})")
    print(f"PCIe Transfer Bandwidth: {budget.h2d_bandwidth_gb_s:.2f} GB/s")

    discovery = DomainDiscovery()
    domains = discovery.create_bootstrap_domains()
    analyzer = ContextAnalyzer(domains)

    print(f"Loading Parameter Map: {map_path} ...")
    param_map = ParameterMap.load(map_path)

    scheduler = BlockScheduler(param_map, hardware_budget=budget, profile=SchedulingProfile.BALANCED)

    # Simulated multi-turn dialogue with topic transitions
    dialogue_turns = [
        ("Turn 1 (Math EN)", "Can you calculate the definite integral of sin(2x) from 0 to pi/2?"),
        ("Turn 2 (Math TR)", "Ayrıca e^(3x) ifadesinin ikinci türevini de adım adım göster."),
        ("Turn 3 (Code Shift)", "Now let's switch topics: implement a thread-safe priority queue in Python."),
        ("Turn 4 (General Shift)", "Thanks a lot! What is a good movie you'd recommend watching this weekend?"),
    ]

    for turn_label, user_input in dialogue_turns:
        print("\n" + "=" * 70)
        print(f"[{turn_label}]")
        print(f"User: \"{user_input}\"")

        # 1. Context Analysis
        analysis = analyzer.analyze(user_input)
        print(f"-> Detected Domain: {analysis.dominant_domain} (Affinity: {analysis.dominant_score:.2%})")
        print(f"-> Topic Drift: {analysis.context_drift:.3f}")

        # 2. Scheduling Decision
        decision = scheduler.schedule(analysis.domain_scores)
        hot_blocks_total = sum(len(b) for b in decision.hot_blocks_per_layer)
        loaded_blocks_total = sum(len(b) for b in decision.loaded_blocks_per_layer)
        evicted_blocks_total = sum(len(b) for b in decision.evicted_blocks_per_layer)

        print(f"-> Active Block Ratio: {decision.active_ratio:.1%}")
        print(f"-> Delta: +{loaded_blocks_total} blocks loaded, -{evicted_blocks_total} blocks evicted")
        print(f"-> Parameter VRAM Saved: {(1.0 - decision.active_ratio) * 100:.1f}%")

    print("\n" + "=" * 70)
    print("Demo completed successfully!")

if __name__ == "__main__":
    main()
