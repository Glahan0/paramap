"""
Parameter Map Construction and Neuron Clustering.
Computes domain selectivity scores, clusters co-activated neurons into blocks,
and serializes universal parameter maps via safetensors.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import json
import torch
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file

@dataclass
class MapMetadata:
    model_name_or_path: str
    block_size: int
    num_layers: int
    intermediate_size: int
    domain_ids: List[str]
    domain_names: Dict[str, str]
    created_at: str

class ParameterMap:
    """
    Universal Parameter Map holding:
    - block_scores: [num_layers, num_blocks, num_domains] in range [0, 1]
    - neuron_order: [num_layers, intermediate_size] permutation mapping original neuron indices to clustered blocks
    - metadata: MapMetadata
    """

    def __init__(
        self,
        block_scores: torch.Tensor,
        neuron_order: torch.Tensor,
        metadata: MapMetadata
    ):
        self.block_scores = block_scores  # [L, NumBlocks, NumDomains]
        self.neuron_order = neuron_order  # [L, IntermediateSize]
        self.metadata = metadata

    @property
    def num_layers(self) -> int:
        return self.metadata.num_layers

    @property
    def num_blocks(self) -> int:
        return self.block_scores.shape[1]

    @property
    def block_size(self) -> int:
        return self.metadata.block_size

    def get_top_blocks_for_domain(
        self,
        domain_idx: int,
        budget_ratio: float = 0.5,
        always_keep_core_ratio: float = 0.1
    ) -> List[torch.Tensor]:
        """
        Returns list of active block indices for each layer given a domain and budget ratio.
        Guarantees that top-core general blocks are always included.
        """
        active_blocks_per_layer = []
        k = max(1, int(self.num_blocks * budget_ratio))
        num_core = max(1, int(self.num_blocks * always_keep_core_ratio))

        # Overall average importance across all domains (core utility)
        core_scores = self.block_scores.mean(dim=-1)  # [L, NumBlocks]

        for layer_idx in range(self.num_layers):
            layer_domain_scores = self.block_scores[layer_idx, :, domain_idx]
            layer_core_scores = core_scores[layer_idx]

            # Top core blocks
            top_core = torch.topk(layer_core_scores, k=num_core).indices

            # Top domain-specific blocks
            top_domain = torch.topk(layer_domain_scores, k=k).indices

            combined = torch.unique(torch.cat([top_core, top_domain]))
            active_blocks_per_layer.append(combined)

        return active_blocks_per_layer

    def save(self, filepath: str) -> None:
        tensors = {
            "block_scores": self.block_scores.contiguous().float(),
            "neuron_order": self.neuron_order.contiguous().long(),
        }
        meta_dict = {
            "model_name_or_path": self.metadata.model_name_or_path,
            "block_size": str(self.metadata.block_size),
            "num_layers": str(self.metadata.num_layers),
            "intermediate_size": str(self.metadata.intermediate_size),
            "domain_ids": json.dumps(self.metadata.domain_ids),
            "domain_names": json.dumps(self.metadata.domain_names),
            "created_at": self.metadata.created_at,
        }
        save_file(tensors, filepath, metadata=meta_dict)

    @classmethod
    def load(cls, filepath: str) -> "ParameterMap":
        """
        Loads a map and restores the metadata `save()` wrote into the safetensors
        header. `domain_ids` in particular must survive the round-trip: the
        scheduler matches incoming domain scores against them, so placeholder ids
        would silently turn block selection context-independent.

        Structural values (layer/block/domain counts) are derived from the tensor
        shapes, which are authoritative; a header that contradicts them means a
        corrupted file and raises.
        """
        with safe_open(filepath, framework="pt") as f:
            header = f.metadata() or {}
            block_scores = f.get_tensor("block_scores")
            neuron_order = f.get_tensor("neuron_order")

        num_layers, num_blocks, num_domains = block_scores.shape
        intermediate_size = neuron_order.shape[1]
        block_size = intermediate_size // num_blocks

        # Validate the header's structural claims against the real shapes.
        for key, actual in (
            ("num_layers", num_layers),
            ("intermediate_size", intermediate_size),
            ("block_size", block_size),
        ):
            if key in header:
                try:
                    claimed = int(header[key])
                except (TypeError, ValueError):
                    raise ValueError(
                        f"{filepath}: header field '{key}' is not an integer: {header[key]!r}"
                    )
                if claimed != actual:
                    raise ValueError(
                        f"{filepath}: header claims {key}={claimed} but tensors imply {actual}."
                    )

        def _decode_json(key: str):
            raw = header.get(key)
            if raw is None:
                return None
            try:
                return json.loads(raw)
            except (TypeError, ValueError):
                return None

        domain_ids = _decode_json("domain_ids")
        if not isinstance(domain_ids, list) or len(domain_ids) != num_domains:
            # Legacy map written without a header. Placeholders keep the object
            # usable, but the caller must assign real ids before scheduling.
            domain_ids = [f"domain_{i}" for i in range(num_domains)]

        domain_names = _decode_json("domain_names")
        if not isinstance(domain_names, dict):
            domain_names = {d_id: d_id for d_id in domain_ids}

        meta = MapMetadata(
            model_name_or_path=header.get("model_name_or_path", "generic"),
            block_size=block_size,
            num_layers=num_layers,
            intermediate_size=intermediate_size,
            domain_ids=domain_ids,
            domain_names=domain_names,
            created_at=header.get("created_at", ""),
        )
        return cls(block_scores, neuron_order, meta)

class MapBuilder:
    """
    Constructs ParameterMap from raw activation profiles across domains.
    """

    def __init__(self, block_size: int = 64):
        self.block_size = block_size

    def build_from_activations(
        self,
        domain_activations: Dict[str, torch.Tensor],  # domain_id -> [L, intermediate_size]
        model_name: str,
    ) -> ParameterMap:
        """
        Takes mean activation magnitudes per domain and builds the clustered parameter map.
        """
        import datetime
        domain_ids = list(domain_activations.keys())
        first_act = next(iter(domain_activations.values()))
        num_layers, intermediate_size = first_act.shape

        num_blocks = intermediate_size // self.block_size
        assert intermediate_size % self.block_size == 0, (
            f"intermediate_size ({intermediate_size}) must be divisible by block_size ({self.block_size})"
        )

        # Stack into [num_domains, L, intermediate_size] -> permute to [L, intermediate_size, num_domains]
        raw_stack = torch.stack([domain_activations[d_id] for d_id in domain_ids], dim=-1) # [L, D_inter, NumDomains]

        all_block_scores = torch.zeros((num_layers, num_blocks, len(domain_ids)), dtype=torch.float32)
        all_neuron_orders = torch.zeros((num_layers, intermediate_size), dtype=torch.long)

        for l in range(num_layers):
            layer_act = raw_stack[l]  # [D_inter, NumDomains]
            
            # Selectivity calculation: Normalize each neuron's domain profile
            # Softmax / ratio across domains
            sums = layer_act.sum(dim=-1, keepdim=True) + 1e-6
            selectivity = layer_act / sums  # [D_inter, NumDomains]

            # Primary domain index for each neuron
            primary_domain = torch.argmax(selectivity, dim=-1)

            # Sort neurons primarily by their primary domain, then by importance
            # This co-locates neurons belonging to the same domain into contiguous blocks!
            sort_keys = primary_domain.float() * 1000.0 + layer_act.max(dim=-1).values
            neuron_order = torch.argsort(sort_keys, descending=True)
            all_neuron_orders[l] = neuron_order

            # Reorder selectivity and activations
            reordered_act = layer_act[neuron_order]  # [D_inter, NumDomains]

            # Pool into blocks
            reshaped = reordered_act.view(num_blocks, self.block_size, len(domain_ids))
            block_means = reshaped.mean(dim=1)  # [num_blocks, NumDomains]

            # Normalize block scores per domain to [0, 1]
            max_vals = block_means.max(dim=0, keepdim=True).values + 1e-6
            normalized_block_scores = block_means / max_vals

            all_block_scores[l] = normalized_block_scores

        meta = MapMetadata(
            model_name_or_path=model_name,
            block_size=self.block_size,
            num_layers=num_layers,
            intermediate_size=intermediate_size,
            domain_ids=domain_ids,
            domain_names={d: d for d in domain_ids},
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

        return ParameterMap(all_block_scores, all_neuron_orders, meta)
