"""
Block-Sparse FFN Inference Engine for ParamMap.
Executes SwiGLU forward passes selectively using active parameter blocks.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

class BlockSparseMLP(nn.Module):
    """
    Drop-in replacement for SwiGLU MLP that executes computation
    only over active parameter blocks.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        block_size: int = 64,
        act_fn: str = "silu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.block_size = block_size
        self.num_blocks = intermediate_size // block_size

        if act_fn == "silu":
            self.act_fn = F.silu
        elif act_fn == "gelu":
            self.act_fn = F.gelu
        else:
            self.act_fn = F.silu

        # Sliced active weights currently resident on device
        self.active_gate: Optional[torch.Tensor] = None   # [num_active_neurons, hidden_size]
        self.active_up: Optional[torch.Tensor] = None     # [num_active_neurons, hidden_size]
        self.active_down: Optional[torch.Tensor] = None   # [hidden_size, num_active_neurons]
        self.active_neuron_count: int = 0

    def load_active_blocks(
        self,
        full_gate_weight: torch.Tensor,
        full_up_weight: torch.Tensor,
        full_down_weight: torch.Tensor,
        active_block_indices: torch.Tensor,
        neuron_permutation: torch.Tensor,
        device: torch.device,
    ) -> None:
        """
        Extracts and loads active parameter blocks into device VRAM.
        `neuron_permutation` maps clustered order to physical weight rows/columns.
        """
        # Determine neuron indices corresponding to active blocks
        active_neurons_list = []
        for b_idx in active_block_indices.tolist():
            start = b_idx * self.block_size
            end = start + self.block_size
            active_neurons_list.append(neuron_permutation[start:end])

        if active_neurons_list:
            active_indices = torch.cat(active_neurons_list).to(device)
            self.active_neuron_count = len(active_indices)

            # Slice weights and move to target device
            self.active_gate = full_gate_weight.index_select(0, active_indices.to(full_gate_weight.device)).to(device)
            self.active_up = full_up_weight.index_select(0, active_indices.to(full_up_weight.device)).to(device)
            self.active_down = full_down_weight.index_select(1, active_indices.to(full_down_weight.device)).to(device)
        else:
            self.active_gate = None
            self.active_up = None
            self.active_down = None
            self.active_neuron_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass over active neurons:
        output = (act_fn(x @ W_gate.T) * (x @ W_up.T)) @ W_down.T
        """
        if self.active_gate is None or self.active_neuron_count == 0:
            return torch.zeros_like(x)

        # x shape: [batch, seq_len, hidden_size]
        gate_out = self.act_fn(F.linear(x, self.active_gate))
        up_out = F.linear(x, self.active_up)
        intermediate = gate_out * up_out
        down_out = F.linear(intermediate, self.active_down)
        return down_out

    def memory_footprint_bytes(self) -> int:
        if self.active_gate is None:
            return 0
        element_size = self.active_gate.element_size()
        total_elements = (
            self.active_gate.numel() +
            self.active_up.numel() +
            self.active_down.numel()
        )
        return total_elements * element_size
