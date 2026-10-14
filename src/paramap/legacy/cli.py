"""
Interactive Command-Line Interface (CLI) for ParamMap.
Provides interactive conversational testing with live telemetry,
multi-lingual support (TR/EN), and real-time scheduling controls.
"""

import sys
import os
import torch

from .engine import ParamMapEngine, SchedulingProfile, EngineStepResult
from .mapper import MapBuilder, ParameterMap
from ..i18n import set_language, get_language, tr

def run_cli():
    print("=" * 70)
    print("                PARAMAP: INTERACTIVE DYNAMIC RUNTIME")
    print("=" * 70)

    # Locate default on-disk GGUF model
    qwen_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf")
    gemma_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/gemma-4-12B-it-QAT-GGUF/gemma-4-12B-it-QAT-Q4_0.gguf")

    model_path = qwen_path if os.path.exists(qwen_path) else gemma_path
    if not os.path.exists(model_path):
        print(f"Error: Could not locate GGUF model on disk at {qwen_path}")
        sys.exit(1)

    print(f"Loaded Target GGUF Model: {os.path.basename(model_path)}")
    
    # Build or load ParameterMap
    map_path = "maps/qwen3.5_9b_param_map.safetensors"
    if os.path.exists(map_path):
        print(f"Loading cached ParameterMap from: {map_path}")
        param_map = ParameterMap.load(map_path)
    else:
        print("Generating structured topology ParameterMap for 32 layers x 192 blocks...")
        builder = MapBuilder(block_size=64)
        num_layers = 32
        inter = 12288
        domain_acts = {
            "math_formal": torch.randn(num_layers, inter).abs(),
            "code_python_system": torch.randn(num_layers, inter).abs(),
            "general_conversation": torch.randn(num_layers, inter).abs(),
            "creative_writing": torch.randn(num_layers, inter).abs(),
        }
        domain_acts["math_formal"][:, :inter//3] *= 3.0
        domain_acts["code_python_system"][:, inter//3:2*inter//3] *= 3.0
        domain_acts["general_conversation"][:, 2*inter//3:] *= 3.0
        param_map = builder.build_from_activations(domain_acts, model_name="Qwen3.5-9B-GGUF")
        os.makedirs("maps", exist_ok=True)
        param_map.save(map_path)

    # Initialize Engine
    engine = ParamMapEngine(
        gguf_model_path=model_path,
        param_map=param_map,
        profile=SchedulingProfile.BALANCED
    )

    print(f"Compute Device: {engine.budget.device_name} ({engine.budget.device_type.upper()})")
    print(f"PCIe Transfer Bandwidth: {engine.budget.h2d_bandwidth_gb_s:.2f} GB/s")
    print(f"Initial Profile: BALANCED (50% target block budget)")
    print("-" * 70)
    print("Commands:")
    print("  /profile [quality|balanced|speed|min_memory]  - Switch dynamic profile")
    print("  /lang [tr|en]                                 - Switch interface language")
    print("  /exit, /quit                                  - Exit application")
    print("-" * 70)

    turn_count = 0
    while True:
        try:
            prompt = input("\n[User] > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting ParamMap CLI.")
            break

        if not prompt:
            continue

        if prompt.lower() in ("/exit", "/quit", "exit", "quit"):
            print("Exiting ParamMap CLI.")
            break

        if prompt.startswith("/profile"):
            parts = prompt.split()
            if len(parts) > 1:
                p_name = parts[1].lower()
                mapping = {
                    "quality": SchedulingProfile.QUALITY,
                    "balanced": SchedulingProfile.BALANCED,
                    "speed": SchedulingProfile.SPEED,
                    "min_memory": SchedulingProfile.MIN_MEMORY,
                }
                if p_name in mapping:
                    engine.set_profile(mapping[p_name])
                    print(f"[*] Profile updated to: {p_name.upper()}")
                else:
                    print(f"[!] Unknown profile. Choose from: {list(mapping.keys())}")
            continue

        if prompt.startswith("/lang"):
            parts = prompt.split()
            if len(parts) > 1:
                set_language(parts[1])
                print(f"[*] Language switched to: {get_language().upper()}")
            continue

        turn_count += 1
        res: EngineStepResult = engine.process_turn(prompt)

        # Print Telemetry Card
        print("-" * 65)
        print(f" Turn #{turn_count} Telemetry")
        print("-" * 65)
        print(f"  * Dominant Domain      : {res.dominant_domain} ({res.dominant_score:.1%})")
        print(f"  * Topic Drift          : {res.context_drift:.3f}")
        print(f"  * Active Blocks (GPU)  : {res.total_active_blocks} ({res.active_block_ratio:.1%})")
        print(f"  * Block Updates        : +{res.loaded_blocks} loaded | -{res.evicted_blocks} evicted")
        print(f"  * PCIe Transfer Time   : {res.transfer_latency_ms:.2f} ms")
        print(f"  * Resident FFN VRAM    : {res.vram_used_mb:.1f} MB ({res.vram_used_mb/1024:.2f} GB)")
        print(f"  * VRAM Saved (Headroom): {res.vram_saved_mb:.1f} MB ({res.vram_saved_mb/1024:.2f} GB / {res.vram_saved_percent:.1f}%)")
        print("-" * 65)

if __name__ == "__main__":
    run_cli()
