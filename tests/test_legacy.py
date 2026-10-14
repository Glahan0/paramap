"""
Tests for the archived domain-conditioned approach (see paramap.legacy).

Kept running so the archived code and its measurements stay verifiable. New
work belongs in test_residency.py.
"""

import os
import tempfile
import warnings
import torch
import pytest
from safetensors.torch import save_file
from paramap.device import DeviceProfiler, HardwareBudget
from paramap.legacy.domains import DomainDiscovery, DomainDefinition
from paramap.legacy.context import ContextAnalyzer
from paramap.legacy.mapper import MapBuilder, ParameterMap
from paramap.legacy.scheduler import BlockScheduler, SchedulingProfile
from paramap.legacy.runtime import BlockSparseMLP

def test_device_profiler():
    profiler = DeviceProfiler()
    mem = profiler.profile_memory()
    assert "ram_total_bytes" in mem
    assert mem["ram_total_bytes"] > 0

    budget = profiler.get_hardware_budget()
    assert isinstance(budget, HardwareBudget)
    assert budget.device_type in ("cuda", "mps", "cpu")

def test_domains_bootstrap_and_pruning():
    discovery = DomainDiscovery()
    domains = discovery.create_bootstrap_domains()
    assert len(domains) >= 4

    # Assign synthetic activation signatures
    domains[0].activation_signature = [1.0, 0.0, 0.0, 0.5]
    domains[1].activation_signature = [0.0, 1.0, 0.5, 0.0]
    domains[2].activation_signature = [0.1, 0.1, 0.9, 0.8]
    domains[3].activation_signature = [0.0, 0.1, 0.8, 0.9]  # Highly correlated with domain 2

    # Pruning & merging test
    result = discovery.validate_and_prune(similarity_threshold=0.90)
    assert result["status"] == "success"

def test_context_analyzer():
    discovery = DomainDiscovery()
    domains = discovery.create_bootstrap_domains()
    analyzer = ContextAnalyzer(domains)

    # Test math prompt
    res_math = analyzer.analyze("Calculate the integral of x^2 + 5x dx = 10.")
    assert "math" in res_math.dominant_domain.lower()

    # Test code prompt
    res_code = analyzer.analyze("def calculate_fibonacci(n):\n    return n if n <= 1 else fib(n-1) + fib(n-2)")
    assert "code" in res_code.dominant_domain.lower()

    # Test novelty detection on unrelated alien text
    res_novel = analyzer.analyze("xyzzy zork blip blop")
    assert res_novel.is_novel_domain is True or res_novel.dominant_score < 0.40

def test_parameter_map_construction_and_io():
    builder = MapBuilder(block_size=64)
    intermediate_size = 256
    num_layers = 4
    num_domains = 3

    # Generate synthetic domain activations: [num_layers, intermediate_size]
    activations = {
        "math": torch.rand(num_layers, intermediate_size),
        "code": torch.rand(num_layers, intermediate_size),
        "general": torch.rand(num_layers, intermediate_size),
    }

    param_map = builder.build_from_activations(activations, model_name="test_model")
    assert param_map.num_layers == num_layers
    assert param_map.num_blocks == 4
    assert param_map.block_size == 64

    # Top blocks retrieval
    top_blocks = param_map.get_top_blocks_for_domain(domain_idx=0, budget_ratio=0.5)
    assert len(top_blocks) == num_layers
    assert len(top_blocks[0]) >= 2

    # Save and load safetensors test
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        param_map.save(tmp_path)
        loaded_map = ParameterMap.load(tmp_path)
        assert loaded_map.num_layers == num_layers
        assert loaded_map.num_blocks == 4
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_block_scheduler_profiles():
    builder = MapBuilder(block_size=32)
    intermediate_size = 128
    num_layers = 2
    activations = {
        "math": torch.rand(num_layers, intermediate_size),
        "code": torch.rand(num_layers, intermediate_size),
    }
    param_map = builder.build_from_activations(activations, model_name="test_model")

    scheduler = BlockScheduler(param_map, profile=SchedulingProfile.BALANCED)
    decision = scheduler.schedule({"math": 0.9, "code": 0.1})

    assert len(decision.hot_blocks_per_layer) == num_layers
    assert decision.active_ratio > 0.3
    assert decision.active_ratio <= 1.0

def test_block_sparse_mlp_forward():
    hidden_size = 64
    intermediate_size = 128
    block_size = 32
    num_blocks = intermediate_size // block_size

    mlp = BlockSparseMLP(hidden_size, intermediate_size, block_size=block_size)

    # Synthetic full weights
    full_gate = torch.randn(intermediate_size, hidden_size)
    full_up = torch.randn(intermediate_size, hidden_size)
    full_down = torch.randn(hidden_size, intermediate_size)

    neuron_order = torch.arange(intermediate_size)
    # Activate first 2 blocks (neurons 0..63)
    active_blocks = torch.tensor([0, 1], dtype=torch.long)

    device = torch.device("cpu")
    mlp.load_active_blocks(full_gate, full_up, full_down, active_blocks, neuron_order, device)

    x = torch.randn(2, 8, hidden_size)  # [batch=2, seq=8, hidden=64]
    out = mlp(x)

    assert out.shape == (2, 8, hidden_size)
    assert not torch.isnan(out).any()
    assert mlp.memory_footprint_bytes() > 0

def test_gguf_model_adapter():
    from paramap.legacy.gguf_adapter import GGUFModelAdapter
    gguf_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf")
    if not os.path.exists(gguf_path):
        pytest.skip("GGUF model file not found on disk")

    adapter = GGUFModelAdapter(gguf_path)
    summary = adapter.get_summary()
    assert summary["num_layers"] == 32
    assert summary["hidden_size"] == 4096
    assert summary["intermediate_size"] == 12288
    assert summary["file_size_gb"] > 5.0

    # Test slicing blocks for layer 0 (activate first 2 blocks)
    active_blocks = torch.tensor([0, 1], dtype=torch.long)
    slices = adapter.slice_mlp_blocks(
        layer_idx=0,
        active_block_indices=active_blocks,
        block_size=64,
        device=torch.device("cpu"),
        dtype=torch.float32
    )
    assert slices["num_active_neurons"] == 128
    assert slices["gate"].shape[0] == 128
    assert slices["gate"].shape[1] == 4096

def test_tiered_memory_manager():
    from paramap.legacy.memory import TieredMemoryManager
    device = torch.device("cpu")
    manager = TieredMemoryManager(device=device)

    # Test warm pinned block registration
    gate = torch.randn(64, 128)
    up = torch.randn(64, 128)
    down = torch.randn(128, 64)
    manager.register_warm_block(layer_idx=0, block_idx=0, gate_slice=gate, up_slice=up, down_slice=down)

    stats = manager.get_stats()
    assert stats.num_warm_blocks == 1

    # Test prefetch to hot
    full_weights = {
        "gate": torch.randn(256, 128),
        "up": torch.randn(256, 128),
        "down": torch.randn(128, 256),
    }
    active_b = torch.tensor([0, 1], dtype=torch.long)
    order = torch.arange(256, dtype=torch.long)
    manager.prefetch_to_hot_async(0, active_b, full_weights, block_size=64, neuron_order=order)
    manager.synchronize()

    hw = manager.get_layer_hot_weights(0)
    assert hw is not None
    assert "gate" in hw
    assert hw["gate"].shape[0] == 128  # 2 blocks * 64

    # Evict
    manager.evict_layer(0)
    assert manager.get_layer_hot_weights(0) is None

def test_paramap_engine():
    from paramap.legacy.engine import ParamMapEngine, SchedulingProfile
    gguf_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf")
    if not os.path.exists(gguf_path):
        pytest.skip("GGUF model file not found on disk")

    # Build synthetic map matching Qwen3.5-9B dimensions
    builder = MapBuilder(block_size=64)
    num_layers = 32
    inter = 12288
    activations = {
        "math_formal": torch.rand(num_layers, inter),
        "code_python_system": torch.rand(num_layers, inter),
    }
    param_map = builder.build_from_activations(activations, model_name="Qwen3.5-9B-GGUF")

    engine = ParamMapEngine(
        gguf_model_path=gguf_path,
        param_map=param_map,
        profile=SchedulingProfile.BALANCED,
        device=torch.device("cpu")
    )

    res = engine.process_turn("Calculate the integral of 5x^4 + 2x dx.")
    assert "math" in res.dominant_domain.lower()
    assert res.active_block_ratio > 0.3
    assert res.vram_saved_percent > 30.0


# ---------------------------------------------------------------------------
# Regression tests: parameter map metadata round-trip
# ---------------------------------------------------------------------------

def _build_domain_biased_map(num_layers=4, intermediate_size=1536, block_size=64):
    """
    Builds a map whose three domains are selective for different neuron ranges,
    so a context-aware scheduler must pick visibly different blocks per domain.
    """
    torch.manual_seed(1234)
    inter = intermediate_size
    activations = {
        "math_formal": torch.rand(num_layers, inter),
        "code_python_system": torch.rand(num_layers, inter),
        "general_conversation": torch.rand(num_layers, inter),
    }
    activations["math_formal"][:, : inter // 3] *= 5.0
    activations["code_python_system"][:, inter // 3 : 2 * inter // 3] *= 5.0
    activations["general_conversation"][:, 2 * inter // 3 :] *= 5.0

    builder = MapBuilder(block_size=block_size)
    return builder.build_from_activations(activations, model_name="Qwen/Test-Model")

def test_parameter_map_metadata_survives_roundtrip(tmp_path):
    param_map = _build_domain_biased_map()
    path = str(tmp_path / "map.safetensors")
    param_map.save(path)
    loaded = ParameterMap.load(path)

    assert loaded.metadata.domain_ids == param_map.metadata.domain_ids
    assert loaded.metadata.domain_names == param_map.metadata.domain_names
    assert loaded.metadata.model_name_or_path == "Qwen/Test-Model"
    assert loaded.metadata.created_at == param_map.metadata.created_at
    assert loaded.metadata.block_size == param_map.metadata.block_size
    assert loaded.metadata.num_layers == param_map.metadata.num_layers
    assert loaded.metadata.intermediate_size == param_map.metadata.intermediate_size
    assert torch.equal(loaded.block_scores, param_map.block_scores)
    assert torch.equal(loaded.neuron_order, param_map.neuron_order)

def test_scheduler_stays_context_sensitive_after_roundtrip(tmp_path):
    """
    Regression guard. ParameterMap.load() used to discard the saved header and
    substitute placeholder domain ids, so BlockScheduler's domain lookup always
    missed, fell back to uniform weights, and returned the same block set for
    every prompt -- silently disabling context-aware scheduling in the CLI and
    the web UI, which both load their map from disk.
    """
    param_map = _build_domain_biased_map()
    path = str(tmp_path / "map.safetensors")
    param_map.save(path)
    loaded = ParameterMap.load(path)

    math_scores = {"math_formal": 0.90, "code_python_system": 0.05, "general_conversation": 0.05}
    code_scores = {"math_formal": 0.05, "code_python_system": 0.90, "general_conversation": 0.05}

    with warnings.catch_warnings():
        # The uniform fallback must not be reached at all for a well-formed map.
        warnings.simplefilter("error", RuntimeWarning)
        math_decision = BlockScheduler(loaded, profile=SchedulingProfile.BALANCED).schedule(math_scores)
        code_decision = BlockScheduler(loaded, profile=SchedulingProfile.BALANCED).schedule(code_scores)

    for layer_idx in range(loaded.num_layers):
        math_blocks = set(math_decision.hot_blocks_per_layer[layer_idx].tolist())
        code_blocks = set(code_decision.hot_blocks_per_layer[layer_idx].tolist())
        assert math_blocks != code_blocks, (
            f"layer {layer_idx} selected identical blocks for math and code: "
            "the scheduler is ignoring domain scores"
        )

def test_scheduler_warns_when_domain_scores_do_not_match_map():
    param_map = _build_domain_biased_map()
    scheduler = BlockScheduler(param_map, profile=SchedulingProfile.BALANCED)

    with pytest.warns(RuntimeWarning, match="uniform, context-independent"):
        scheduler.schedule({"domain_0": 0.9, "domain_1": 0.1})

def test_parameter_map_load_rejects_contradictory_header(tmp_path):
    param_map = _build_domain_biased_map()
    path = str(tmp_path / "corrupt.safetensors")
    save_file(
        {
            "block_scores": param_map.block_scores.contiguous(),
            "neuron_order": param_map.neuron_order.contiguous(),
        },
        path,
        metadata={"num_layers": "99"},
    )

    with pytest.raises(ValueError, match="num_layers"):
        ParameterMap.load(path)

def test_parameter_map_load_tolerates_headerless_legacy_map(tmp_path):
    param_map = _build_domain_biased_map()
    path = str(tmp_path / "legacy.safetensors")
    save_file(
        {
            "block_scores": param_map.block_scores.contiguous(),
            "neuron_order": param_map.neuron_order.contiguous(),
        },
        path,
    )

    loaded = ParameterMap.load(path)
    assert loaded.metadata.domain_ids == ["domain_0", "domain_1", "domain_2"]
    assert loaded.metadata.domain_names == {d: d for d in loaded.metadata.domain_ids}
    assert loaded.num_layers == param_map.num_layers
    assert loaded.block_size == param_map.block_size


# ---------------------------------------------------------------------------
# Evaluation harness tests
# ---------------------------------------------------------------------------

class _TinyMLP(torch.nn.Module):
    """Minimal SwiGLU MLP with the attribute names the Qwen adapter expects."""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        import torch.nn.functional as F
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class _TinyConfig:
    def __init__(self, hidden_size, intermediate_size):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = "silu"

class _TinyModel(torch.nn.Module):
    """Stands in for a causal LM so the eval harness can be tested offline."""

    def __init__(self, num_layers=2, hidden_size=32, intermediate_size=256):
        super().__init__()
        layer_list = []
        for _ in range(num_layers):
            layer = torch.nn.Module()
            layer.mlp = _TinyMLP(hidden_size, intermediate_size)
            layer_list.append(layer)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(layer_list)
        self.config = _TinyConfig(hidden_size, intermediate_size)

def _tiny_map(num_layers=2, intermediate_size=256, block_size=64):
    torch.manual_seed(7)
    activations = {
        "math_formal": torch.rand(num_layers, intermediate_size),
        "code_python_system": torch.rand(num_layers, intermediate_size),
    }
    activations["math_formal"][:, : intermediate_size // 2] *= 4.0
    activations["code_python_system"][:, intermediate_size // 2 :] *= 4.0
    return MapBuilder(block_size=block_size).build_from_activations(activations, model_name="tiny")

def test_eval_strategies_are_budget_matched():
    """
    The original feasibility experiment compared a core-union selection against
    exactly-50% random selection, so the two sides could differ in how many
    blocks were active. Every strategy here must return exactly k.
    """
    from paramap.legacy.eval import (
        blocks_for_budget,
        select_importance,
        select_paramap,
        select_paramap_with_core,
        select_random,
    )

    param_map = _tiny_map()
    k = blocks_for_budget(param_map, 0.5)
    assert k == param_map.num_blocks // 2

    selections = {
        "paramap": select_paramap(param_map, 0, k),
        "paramap_core": select_paramap_with_core(param_map, 0, k),
        "importance": select_importance(param_map, k),
        "random": select_random(param_map, k),
    }

    for name, per_layer in selections.items():
        assert len(per_layer) == param_map.num_layers, name
        for layer_idx, blocks in enumerate(per_layer):
            assert blocks.numel() == k, f"{name} layer {layer_idx}: {blocks.numel()} != {k}"
            assert len(set(blocks.tolist())) == k, f"{name} layer {layer_idx} has duplicates"
            assert int(blocks.min()) >= 0 and int(blocks.max()) < param_map.num_blocks, name

def test_eval_masked_and_sliced_paths_agree():
    """
    Zeroing inactive neurons at the down-projection input and physically slicing
    them away are mathematically the same operation. If these drift apart, one of
    the two paths indexes the wrong axis or applies the wrong permutation.
    """
    from paramap.legacy.eval import blocks_for_budget, masked_blocks, select_paramap, sliced_blocks

    param_map = _tiny_map()
    model = _TinyModel(num_layers=param_map.num_layers, intermediate_size=param_map.metadata.intermediate_size)
    model.eval()

    blocks = select_paramap(param_map, 0, blocks_for_budget(param_map, 0.5))
    x = torch.randn(1, 5, model.config.hidden_size)

    with torch.no_grad():
        with masked_blocks(model, param_map, blocks):
            masked_out = model.model.layers[0].mlp(x)
        with sliced_blocks(model, param_map, blocks):
            sliced_out = model.model.layers[0].mlp(x)

    assert masked_out.shape == sliced_out.shape
    assert torch.allclose(masked_out, sliced_out, rtol=1e-4, atol=1e-4)

def test_eval_context_managers_restore_the_model():
    from paramap.legacy.eval import blocks_for_budget, masked_blocks, select_paramap, sliced_blocks

    param_map = _tiny_map()
    model = _TinyModel(num_layers=param_map.num_layers, intermediate_size=param_map.metadata.intermediate_size)
    model.eval()

    original_mlps = [model.model.layers[i].mlp for i in range(param_map.num_layers)]
    blocks = select_paramap(param_map, 0, blocks_for_budget(param_map, 0.5))
    x = torch.randn(1, 3, model.config.hidden_size)

    with torch.no_grad():
        reference = model.model.layers[0].mlp(x)

        with sliced_blocks(model, param_map, blocks):
            pass
        assert [model.model.layers[i].mlp for i in range(param_map.num_layers)] == original_mlps

        with masked_blocks(model, param_map, blocks):
            pass
        restored = model.model.layers[0].mlp(x)

    assert torch.allclose(reference, restored), "masking hooks were not removed"

def test_eval_select_rejects_domain_absent_from_map():
    from paramap.legacy.eval import ParamMapEvaluator

    param_map = _tiny_map()
    model = _TinyModel(num_layers=param_map.num_layers, intermediate_size=param_map.metadata.intermediate_size)
    evaluator = ParamMapEvaluator(model, tokenizer=None, param_map=param_map, device=torch.device("cpu"))

    with pytest.raises(ValueError, match="not in this map"):
        evaluator.select("paramap", "no_such_domain", k=2)

def test_eval_evaluator_rejects_mismatched_model():
    from paramap.legacy.eval import ParamMapEvaluator

    param_map = _tiny_map(num_layers=2)
    model = _TinyModel(num_layers=3, intermediate_size=param_map.metadata.intermediate_size)

    with pytest.raises(ValueError, match="not profiled from the same architecture"):
        ParamMapEvaluator(model, tokenizer=None, param_map=param_map, device=torch.device("cpu"))

def test_eval_gate_report_flags_a_losing_paramap():
    """check_gates must not claim success when the baseline is better."""
    from paramap.legacy.eval import QualityRow, ThroughputRow, check_gates

    quality = [
        QualityRow("math_formal", "paramap_core", 0.5, 2, 20.0, 20.0),
        QualityRow("math_formal", "importance", 0.5, 2, 15.0, 10.0),
        QualityRow("math_formal", "random", 0.5, 2, 40.0, 60.0),
    ]
    throughput = [ThroughputRow("sliced", 0.5, 2, 90.0, -25.0)]

    gates = check_gates(quality, throughput, budget_ratio=0.5)
    assert gates["beats_random"] is True
    assert gates["beats_importance_baseline"] is False
    assert gates["quality_gate_passed"] is False
    assert gates["throughput_gate_passed"] is False
    assert gates["domains_won_vs_importance"] == "0/1"
