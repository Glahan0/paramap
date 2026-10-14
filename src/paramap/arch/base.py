"""
Base Architecture Adapter for Gated FFN / SwiGLU LLMs.
Abstracts model-specific layer access, neuron indexing, and weight slicing.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn

class ModelArchAdapter(ABC):
    """
    Abstract adapter for extracting intermediate activation statistics,
    identifying FFN dimensions, and slicing weight matrices into blocks.
    """

    @abstractmethod
    def get_layers(self, model: nn.Module) -> nn.ModuleList:
        """
        Returns the container holding the decoder layers, so callers can address
        layers as a group without reaching into model internals.
        """
        pass

    @abstractmethod
    def get_num_layers(self, model: nn.Module) -> int:
        """Returns total number of transformer layers."""
        pass

    @abstractmethod
    def get_intermediate_size(self, model: nn.Module) -> int:
        """Returns intermediate (FFN) dimension size."""
        pass

    @abstractmethod
    def get_hidden_size(self, model: nn.Module) -> int:
        """Returns hidden dimension size."""
        pass

    @abstractmethod
    def get_layer_mlp(self, model: nn.Module, layer_idx: int) -> nn.Module:
        """Returns MLP / FFN submodule for layer layer_idx."""
        pass

    @abstractmethod
    def set_layer_mlp(self, model: nn.Module, layer_idx: int, mlp: nn.Module) -> None:
        """
        Replaces the MLP / FFN submodule of layer `layer_idx`. Needed to swap in a
        block-sparse FFN, so callers never have to reach into model internals.
        """
        pass

    @abstractmethod
    def register_activation_hook(
        self,
        model: nn.Module,
        layer_idx: int,
        hook_fn: Any
    ) -> Any:
        """
        Registers forward hook on intermediate activation (e.g. act_fn(gate) * up).
        Returns hook handle.
        """
        pass

    @abstractmethod
    def slice_mlp_weights(
        self,
        mlp_module: nn.Module,
        neuron_indices: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Extracts sliced gate, up, and down projections for given neuron indices.
        """
        pass
