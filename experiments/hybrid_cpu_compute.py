"""
Measures the alternative to streaming: compute the decoder layers that do not
fit in VRAM on the CPU, from RAM, instead of moving their weights over PCIe.

First k layers run on the GPU, the rest on the CPU; hidden states cross the bus
instead of weights. Reports tokens/s, peak VRAM and how many generated tokens
match the all-GPU baseline (CPU arithmetic is not bit-identical to the GPU's).

    python experiments/hybrid_cpu_compute.py 27 20 13 6 3 0
"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from paramap.benchmark import run_generation, DEFAULT_PROMPT

def move(x, dev):
    if torch.is_tensor(x): return x.to(dev)
    if isinstance(x, tuple): return tuple(move(v, dev) for v in x)
    if isinstance(x, list): return [move(v, dev) for v in x]
    if isinstance(x, dict): return {k: move(v, dev) for k, v in x.items()}
    return x

name = "Qwen/Qwen2.5-3B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).eval()
gpu, cpu = torch.device("cuda"), torch.device("cpu")
layers = m.model.layers; L = len(layers)
torch.set_num_threads(6)

# Baseline: everything on GPU.
m.to(gpu)
base_ids, _, base_tps, base_peak = run_generation(m, tok, DEFAULT_PROMPT, gpu, max_new_tokens=48)
print(f"baseline   {L:2d}G/ 0C  {base_tps:5.1f} tok/s  peak {base_peak/1e9:.2f} GB", flush=True)

handles = []
for k in [int(a) for a in sys.argv[1:]]:
    for h in handles: h.remove()
    handles = []
    for i, layer in enumerate(layers):
        layer.to(gpu if i < k else cpu)
    torch.cuda.empty_cache()
    for i in range(k, L):
        handles.append(layers[i].register_forward_pre_hook(
            lambda mod, args, kwargs: (move(args, cpu), move(kwargs, cpu)), with_kwargs=True))
    handles.append(layers[L-1].register_forward_hook(lambda mod, args, out: move(out, gpu)))
    ids, _, tps, peak = run_generation(m, tok, DEFAULT_PROMPT, gpu, max_new_tokens=48)
    same = next((j for j, (a, b) in enumerate(zip(ids, base_ids)) if a != b), len(ids))
    print(f"hybrid     {k:2d}G/{L-k:2d}C  {tps:5.1f} tok/s  peak {peak/1e9:.2f} GB  "
          f"({tps/base_tps-1:+.1%})  identical tokens: {same}/48", flush=True)
