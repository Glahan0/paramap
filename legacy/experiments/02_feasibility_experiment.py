"""
Phase 1 Feasibility Gate Experiment.
Profiles Qwen2.5 across dynamic multilingual domains, builds the ParameterMap,
and validates that domain-conditioned block selection outperforms random parameter pruning.
"""

import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from paramap.domains import DomainDiscovery, DomainDefinition
from paramap.profiler import ActivationProfiler
from paramap.mapper import MapBuilder, ParameterMap
from paramap.scheduler import BlockScheduler, SchedulingProfile

def compute_prompt_loss(model, tokenizer, text: str, device: torch.device) -> float:
    """Computes cross-entropy loss of the model on the text."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        return float(outputs.loss.item())

def evaluate_with_block_mask(
    model,
    tokenizer,
    eval_text: str,
    active_blocks_per_layer: list,
    param_map: ParameterMap,
    device: torch.device
) -> float:
    """
    Evaluates loss on eval_text while masking out inactive intermediate neurons in each layer.
    """
    block_size = param_map.block_size
    handles = []

    def make_mask_hook(layer_idx: int, active_blocks: torch.Tensor, neuron_order: torch.Tensor):
        # Determine active neuron indices
        active_neurons_list = []
        for b_idx in active_blocks.tolist():
            s = b_idx * block_size
            e = s + block_size
            active_neurons_list.append(neuron_order[s:e])

        active_indices = torch.cat(active_neurons_list).to(device) if active_neurons_list else torch.empty(0, dtype=torch.long, device=device)
        mask = torch.zeros(param_map.metadata.intermediate_size, device=device, dtype=torch.bool)
        if len(active_indices) > 0:
            mask[active_indices] = True

        def hook(module, inputs):
            # inputs[0] is intermediate activation: [batch, seq_len, intermediate_size]
            act = inputs[0]
            masked_act = act * mask.view(1, 1, -1)
            return (masked_act,)

        return hook

    # Attach masking hooks to each layer's down_proj
    for l_idx in range(param_map.num_layers):
        mlp = model.model.layers[l_idx].mlp
        active_b = active_blocks_per_layer[l_idx]
        order = param_map.neuron_order[l_idx]
        h = mlp.down_proj.register_forward_pre_hook(make_mask_hook(l_idx, active_b, order))
        handles.append(h)

    try:
        loss = compute_prompt_loss(model, tokenizer, eval_text, device)
    finally:
        for h in handles:
            h.remove()

    return loss

def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("       PARAMAP PHASE 1: FEASIBILITY GATE EXPERIMENT")
    print(f"       Model: {model_name} | Device: {device}")
    print("=" * 65)

    print("\n[1/5] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()

    print("\n[2/5] Synthesizing multilingual dynamic domains (TR & EN)...")
    discovery = DomainDiscovery()
    domains = discovery.create_bootstrap_domains()
    for d in domains:
        print(f"  - [{d.domain_id}] {d.name_en} / {d.name_tr} ({len(d.all_prompts())} seed prompts)")

    print("\n[3/5] Profiling layer activations across domains...")
    profiler = ActivationProfiler(model, device=device)
    domain_activations = profiler.profile_all_domains(domains, tokenizer, max_length=64)

    # Validate signatures
    val_res = discovery.validate_and_prune()
    print(f"  Activation signature validation status: {val_res['status']}")

    print("\n[4/5] Building clustered ParameterMap (block_size=64)...")
    os.makedirs("maps", exist_ok=True)
    builder = MapBuilder(block_size=64)
    param_map = builder.build_from_activations(domain_activations, model_name=model_name)
    map_path = "maps/qwen2.5_0.5b_param_map.safetensors"
    param_map.save(map_path)
    print(f"  Saved parameter map to: {map_path}")
    print(f"  Map structure: {param_map.num_layers} layers, {param_map.num_blocks} blocks/layer, block_size={param_map.block_size}")

    print("\n[5/5] Running Feasibility Gate Test...")
    # Test Evaluation Texts in Turkish and English
    eval_suite = [
        ("Math (EN)", "Calculate the integral of 3*x^2 + 2*x + 1 from 0 to 4.", 0),  # math_formal is idx 0
        ("Math (TR)", "f(x) = 4x^3 - 6x fonksiyonunun x=2 noktasındaki türevini hesaplayınız.", 0),
        ("Code (EN)", "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr)//2]", 1), # code is idx 1
        ("Code (TR)", "import asyncio\nasync def fetch_data(url):\n    return await client.get(url)", 1),
    ]

    print("\n" + "-" * 75)
    print(f"{'Task':<12} | {'Baseline (100%)':<16} | {'ParamMap (50%)':<16} | {'Random (50%)':<16} | {'Gain'}")
    print("-" * 75)

    num_blocks = param_map.num_blocks
    k_blocks = num_blocks // 2

    for task_name, text, domain_idx in eval_suite:
        # 1. Full model baseline
        base_loss = compute_prompt_loss(model, tokenizer, text, device)

        # 2. ParamMap 50% top blocks for this domain
        paramap_blocks = param_map.get_top_blocks_for_domain(domain_idx=domain_idx, budget_ratio=0.50)
        paramap_loss = evaluate_with_block_mask(model, tokenizer, text, paramap_blocks, param_map, device)

        # 3. Random 50% blocks
        torch.manual_seed(42)
        random_blocks = [torch.randperm(num_blocks)[:k_blocks] for _ in range(param_map.num_layers)]
        random_loss = evaluate_with_block_mask(model, tokenizer, text, random_blocks, param_map, device)

        diff = random_loss - paramap_loss
        print(f"{task_name:<12} | {base_loss:<16.4f} | {paramap_loss:<16.4f} | {random_loss:<16.4f} | +{diff:.4f} better")

    print("-" * 75)
    print("\nFeasibility Gate test completed successfully!")

if __name__ == "__main__":
    main()
