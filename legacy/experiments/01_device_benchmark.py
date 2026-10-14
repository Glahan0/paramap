"""
Hardware Benchmark and Memory Budgeting Script.
"""

from paramap.device import DeviceProfiler

def main():
    print("=" * 60)
    print("           PARAMAP HARDWARE & DEVICE BENCHMARK")
    print("=" * 60)

    profiler = DeviceProfiler()
    print(f"Detected compute device: {profiler.device_type}")

    mem = profiler.profile_memory()
    print(f"GPU Name: {mem.get('gpu_name')}")
    print(f"GPU VRAM Free: {mem.get('vram_free_bytes', 0) / (1024**3):.2f} GB / {mem.get('vram_total_bytes', 0) / (1024**3):.2f} GB")
    print(f"System RAM Free: {mem.get('ram_free_bytes', 0) / (1024**3):.2f} GB / {mem.get('ram_total_bytes', 0) / (1024**3):.2f} GB")

    print("\nMeasuring PCIe Transfer Bandwidth (128 MB pinned memory)...")
    bandwidth = profiler.benchmark_bandwidth(test_size_mb=128)
    print(f"  Host -> Device (RAM to VRAM): {bandwidth['h2d_gb_s']:.2f} GB/s")
    print(f"  Device -> Host (VRAM to RAM): {bandwidth['d2h_gb_s']:.2f} GB/s")

    budget = profiler.get_hardware_budget()
    print("\nRecommended Dynamic Budget:")
    print(f"  Max Recommended Hot VRAM for Weights: {budget.recommended_hot_vram_bytes / (1024**3):.2f} GB")
    print(f"  Max Recommended Warm RAM for Weights: {budget.recommended_warm_ram_bytes / (1024**3):.2f} GB")
    print("=" * 60)

if __name__ == "__main__":
    main()
