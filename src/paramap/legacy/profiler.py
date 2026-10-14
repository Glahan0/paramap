"""
Layer-Wise Streaming Activation Profiler for ParamMap.
Captures intermediate SwiGLU neuron activations across domain corpora
with low memory overhead, supporting both full-model and layer-streaming modes.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
from tqdm import tqdm
from ..arch.base import ModelArchAdapter
from ..arch.qwen import QwenArchitectureAdapter
from .domains import DomainDefinition

class ActivationProfiler:
    """
    Profiles intermediate neuron activations for each transformer layer across domains.
    """

    def __init__(
        self,
        model: nn.Module,
        adapter: Optional[ModelArchAdapter] = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.adapter = adapter or QwenArchitectureAdapter()
        self.device = device or next(model.parameters()).device
        self.num_layers = self.adapter.get_num_layers(model)
        self.intermediate_size = self.adapter.get_intermediate_size(model)

    def profile_domain(
        self,
        domain: DomainDefinition,
        tokenizer: Any,
        max_length: int = 128,
        batch_size: int = 2
    ) -> torch.Tensor:
        """
        Runs domain prompts through the model and computes the mean activation
        vector for each layer: returns Tensor of shape [num_layers, intermediate_size].
        """
        prompts = domain.all_prompts()
        if not prompts:
            raise ValueError(f"Domain '{domain.domain_id}' contains no seed prompts.")

        # Accumulator: [num_layers, intermediate_size] on CPU
        total_activations = torch.zeros((self.num_layers, self.intermediate_size), dtype=torch.float32)
        total_tokens = [0] * self.num_layers

        # Buffer for current batch
        layer_batch_acts: Dict[int, List[torch.Tensor]] = {l: [] for l in range(self.num_layers)}

        def make_hook(layer_idx: int):
            def hook_fn(l_idx: int, intermediate_act: torch.Tensor):
                # intermediate_act shape: [batch, seq_len, intermediate_size]
                # We compute absolute magnitude and average across sequence dimension
                with torch.no_grad():
                    act_cpu = intermediate_act.detach().abs().float().cpu()
                    # Reshape to [batch * seq_len, intermediate_size]
                    flat = act_cpu.view(-1, self.intermediate_size)
                    layer_batch_acts[layer_idx].append(flat)
            return hook_fn

        # Attach hooks
        handles = []
        for l in range(self.num_layers):
            h = self.adapter.register_activation_hook(self.model, l, make_hook(l))
            handles.append(h)

        self.model.eval()

        try:
            with torch.no_grad():
                for i in range(0, len(prompts), batch_size):
                    batch_texts = prompts[i : i + batch_size]
                    encoded = tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt"
                    )
                    input_ids = encoded["input_ids"].to(self.device)
                    attention_mask = encoded.get("attention_mask", None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)

                    # Forward pass
                    _ = self.model(input_ids=input_ids, attention_mask=attention_mask)

                    # Drain collected batch activations
                    for l in range(self.num_layers):
                        if layer_batch_acts[l]:
                            stacked = torch.cat(layer_batch_acts[l], dim=0)
                            total_activations[l] += stacked.sum(dim=0)
                            total_tokens[l] += stacked.shape[0]
                            layer_batch_acts[l].clear()

        finally:
            # Always remove hooks cleanly
            for h in handles:
                h.remove()

        # Compute mean
        for l in range(self.num_layers):
            if total_tokens[l] > 0:
                total_activations[l] /= total_tokens[l]

        return total_activations

    def profile_all_domains(
        self,
        domains: List[DomainDefinition],
        tokenizer: Any,
        max_length: int = 128
    ) -> Dict[str, torch.Tensor]:
        """
        Profiles all domains and stores activation signatures into DomainDefinition objects.
        Returns mapping from domain_id to activation tensor [num_layers, intermediate_size].
        """
        results: Dict[str, torch.Tensor] = {}

        for domain in domains:
            print(f"--> Profiling domain: {domain.name_en} ({domain.domain_id})")
            act = self.profile_domain(domain, tokenizer, max_length=max_length)
            results[domain.domain_id] = act

            # Summarize domain signature as layer-averaged neuron vector for signature comparison
            signature = act.mean(dim=0).tolist()
            domain.activation_signature = signature

        return results
