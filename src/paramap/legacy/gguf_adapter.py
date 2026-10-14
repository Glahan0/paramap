"""
GGUF Architecture Adapter and Quantized Tensor Slicer for ParamMap.
Directly reads GGUF models on disk (e.g. Qwen3.5-9B, Gemma-4-12B), extracts
intermediate dimensions, and dequantizes/slices active parameter blocks on demand.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import gguf
from gguf import quants

class GGUFModelAdapter:
    """
    Adapter for reading, analyzing, and slicing GGUF quantized models on disk.
    Supports Q4_K, Q6_K, Q4_0, Q8_0 and standard floating point types.
    """

    def __init__(self, gguf_path: str):
        self.gguf_path = os.path.expanduser(gguf_path)
        if not os.path.exists(self.gguf_path):
            raise FileNotFoundError(f"GGUF model file not found: {self.gguf_path}")

        self.reader = gguf.GGUFReader(self.gguf_path)
        self.arch = bytes(self.reader.fields["general.architecture"].parts[-1]).decode("utf-8")
        
        # Metadata
        self.num_layers = self._get_meta_int(f"{self.arch}.block_count", default=32)
        self.hidden_size = self._get_meta_int(f"{self.arch}.embedding_length", default=4096)
        self.intermediate_size = self._get_meta_int(f"{self.arch}.feed_forward_length", default=12288)

        # Build tensor index map
        self._tensor_map: Dict[str, Any] = {t.name: t for t in self.reader.tensors}

    def _get_meta_int(self, key: str, default: int) -> int:
        field = self.reader.fields.get(key)
        if field and len(field.parts) > 0:
            try:
                return int(field.parts[-1][0])
            except Exception:
                pass
        return default

    def get_summary(self) -> Dict[str, Any]:
        return {
            "model_path": self.gguf_path,
            "architecture": self.arch,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "file_size_bytes": os.path.getsize(self.gguf_path),
            "file_size_gb": round(os.path.getsize(self.gguf_path) / (1024**3), 2),
            "tensor_count": len(self.reader.tensors),
        }

    def get_layer_mlp_tensors(self, layer_idx: int) -> Dict[str, Any]:
        """
        Locates the gate, up, and down projection tensors for a given layer.
        """
        gate_name = f"blk.{layer_idx}.ffn_gate.weight"
        up_name = f"blk.{layer_idx}.ffn_up.weight"
        down_name = f"blk.{layer_idx}.ffn_down.weight"

        return {
            "gate": self._tensor_map.get(gate_name),
            "up": self._tensor_map.get(up_name),
            "down": self._tensor_map.get(down_name),
        }

    def dequantize_tensor(
        self,
        tensor_entry: Any,
        target_dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """
        Dequantizes a GGUF tensor into a PyTorch tensor.
        """
        raw_data = tensor_entry.data
        t_type = tensor_entry.tensor_type

        # Use official gguf dequantize routine
        arr = quants.dequantize(raw_data, t_type)
        t = torch.from_numpy(arr).to(dtype=target_dtype, device=device)
        return t

    def slice_mlp_blocks(
        self,
        layer_idx: int,
        active_block_indices: torch.Tensor,
        block_size: int,
        neuron_order: Optional[torch.Tensor] = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype: torch.dtype = torch.float16
    ) -> Dict[str, torch.Tensor]:
        """
        Extracts only active blocks for layer_idx, dequantizes them, and moves to device.
        """
        mlp_tensors = self.get_layer_mlp_tensors(layer_idx)
        if not all(mlp_tensors.values()):
            raise ValueError(f"Missing FFN tensors for layer {layer_idx}")

        # Dequantize full layer MLP on CPU
        gate_w = self.dequantize_tensor(mlp_tensors["gate"], target_dtype=dtype)
        up_w = self.dequantize_tensor(mlp_tensors["up"], target_dtype=dtype)
        down_w = self.dequantize_tensor(mlp_tensors["down"], target_dtype=dtype)

        # Compute constituent neuron indices
        if neuron_order is None:
            neuron_order = torch.arange(self.intermediate_size, dtype=torch.long)

        active_neurons_list = []
        for b_idx in active_block_indices.tolist():
            s = b_idx * block_size
            e = s + block_size
            active_neurons_list.append(neuron_order[s:e])

        if active_neurons_list:
            active_indices = torch.cat(active_neurons_list)
            # Slices:
            # gate: [intermediate_size, hidden_size] -> select row
            # up: [intermediate_size, hidden_size] -> select row
            # down: [hidden_size, intermediate_size] -> select col (or vice versa depending on layout)
            gate_slice = gate_w.index_select(0, active_indices).to(device)
            up_slice = up_w.index_select(0, active_indices).to(device)

            # In GGUF, down_proj shape is [hidden_size, intermediate_size] or [intermediate_size, hidden_size]
            if down_w.shape[1] == self.intermediate_size:
                down_slice = down_w.index_select(1, active_indices).to(device)
            else:
                down_slice = down_w.index_select(0, active_indices).to(device)
        else:
            gate_slice = torch.empty(0, dtype=dtype, device=device)
            up_slice = torch.empty(0, dtype=dtype, device=device)
            down_slice = torch.empty(0, dtype=dtype, device=device)

        return {
            "gate": gate_slice,
            "up": up_slice,
            "down": down_slice,
            "num_active_neurons": len(active_indices) if active_neurons_list else 0,
        }
