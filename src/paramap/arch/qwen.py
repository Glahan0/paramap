"""
Qwen2 / Qwen2.5 (and Llama/Mistral) SwiGLU Architecture Adapter.
"""

from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
from .base import ModelArchAdapter

class QwenArchitectureAdapter(ModelArchAdapter):
    """
    Adapter for Qwen2 / Qwen2.5 and compatible Llama-style SwiGLU models.
    """

    def get_layers(self, model: nn.Module) -> nn.ModuleList:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        elif hasattr(model, "layers"):
            return model.layers
        raise AttributeError("Could not find layers in model structure.")

    def get_num_layers(self, model: nn.Module) -> int:
        return len(self.get_layers(model))

    def get_intermediate_size(self, model: nn.Module) -> int:
        config = getattr(model, "config", None)
        if config and hasattr(config, "intermediate_size"):
            return config.intermediate_size
        mlp = self.get_layer_mlp(model, 0)
        return mlp.gate_proj.out_features

    def get_hidden_size(self, model: nn.Module) -> int:
        config = getattr(model, "config", None)
        if config and hasattr(config, "hidden_size"):
            return config.hidden_size
        mlp = self.get_layer_mlp(model, 0)
        return mlp.gate_proj.in_features

    def get_layer_mlp(self, model: nn.Module, layer_idx: int) -> nn.Module:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers[layer_idx].mlp
        elif hasattr(model, "layers"):
            return model.layers[layer_idx].mlp
        raise AttributeError(f"Could not locate layer {layer_idx} MLP.")

    def set_layer_mlp(self, model: nn.Module, layer_idx: int, mlp: nn.Module) -> None:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            model.model.layers[layer_idx].mlp = mlp
            return
        elif hasattr(model, "layers"):
            model.layers[layer_idx].mlp = mlp
            return
        raise AttributeError(f"Could not locate layer {layer_idx} to replace its MLP.")

    def register_activation_hook(
        self,
        model: nn.Module,
        layer_idx: int,
        hook_fn: Any
    ) -> Any:
        """
        Attaches a hook to capture SwiGLU intermediate activation:
        activations = act_fn(gate_proj(x)) * up_proj(x)
        """
        mlp = self.get_layer_mlp(model, layer_idx)

        # In transformers Qwen2MLP / LlamaMLP:
        # forward(x) computes: return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        # Hooking onto down_proj input directly gives intermediate activation!
        def down_proj_pre_hook(module, inputs):
            intermediate_act = inputs[0]  # shape: [batch, seq_len, intermediate_size]
            hook_fn(layer_idx, intermediate_act)

        handle = mlp.down_proj.register_forward_pre_hook(down_proj_pre_hook)
        return handle

    def slice_mlp_weights(
        self,
        mlp_module: nn.Module,
        neuron_indices: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Slices gate_proj, up_proj rows and down_proj columns for specified neuron indices.
        """
        indices = neuron_indices.to(mlp_module.gate_proj.weight.device)
        return {
            "gate_proj": mlp_module.gate_proj.weight.index_select(0, indices).detach().clone(),
            "up_proj": mlp_module.up_proj.weight.index_select(0, indices).detach().clone(),
            "down_proj": mlp_module.down_proj.weight.index_select(1, indices).detach().clone(),
        }
