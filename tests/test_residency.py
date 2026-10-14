"""
Tests for budgeted, lossless VRAM/RAM residency.

The invariant that matters most runs on CPU and needs no large model: streaming
a layer's weights in just before it executes must not change the result. If
`test_streaming_output_is_identical` ever fails, the runtime is no longer
lossless and nothing else about it is worth measuring.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from paramap.residency import (
    InsufficientVRAMError,
    ResidencyPlan,
    StreamingModel,
    estimate_kv_cache_bytes,
    format_bytes,
    module_bytes,
    parse_bytes,
    plan_residency,
)


# ---------------------------------------------------------------------------
# A runnable stand-in with the layer layout the adapters expect
# ---------------------------------------------------------------------------

class _TinyBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        return x + self.fc2(torch.relu(self.fc1(x)))


class _TinyInner(nn.Module):
    def __init__(self, num_layers, hidden_size, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([_TinyBlock(hidden_size) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class _TinyLM(nn.Module):
    def __init__(self, num_layers=6, hidden_size=16, vocab_size=32):
        super().__init__()
        self.model = _TinyInner(num_layers, hidden_size, vocab_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = SimpleNamespace(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=2,
            num_key_value_heads=1,
        )

    def forward(self, input_ids):
        return self.lm_head(self.model(input_ids))


def _plan_for(model, resident_count, bandwidth_gb_s=10.0):
    """Builds a plan directly so a test can pin the resident/streamed split."""
    layers = model.model.layers
    layer_bytes = max(module_bytes(layer) for layer in layers)
    anchor_bytes = module_bytes(model) - sum(module_bytes(layer) for layer in layers)
    num_layers = len(layers)
    return ResidencyPlan(
        num_layers=num_layers,
        layer_bytes=layer_bytes,
        anchor_bytes=anchor_bytes,
        vram_budget_bytes=anchor_bytes + (resident_count + 2) * layer_bytes,
        kv_cache_reserve_bytes=0,
        activation_reserve_bytes=0,
        staging_slots=2,
        resident_layers=list(range(resident_count)),
        streamed_layers=list(range(resident_count, num_layers)),
        bandwidth_gb_s=bandwidth_gb_s,
        dtype_bytes=4,
    )


# ---------------------------------------------------------------------------
# Byte parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("1024", 1024),
        ("1kb", 1024),
        ("1KB", 1024),
        ("4GB", 4 * 1024**3),
        ("512 MB", 512 * 1024**2),
        ("3.5g", int(3.5 * 1024**3)),
        (2048, 2048),
    ],
)
def test_parse_bytes_accepts_human_sizes(text, expected):
    assert parse_bytes(text) == expected


def test_parse_bytes_rejects_nonsense():
    with pytest.raises(ValueError):
        parse_bytes("plenty")
    with pytest.raises(ValueError):
        parse_bytes("10 parsecs")


def test_format_bytes_picks_a_sensible_unit():
    assert format_bytes(4 * 1024**3).endswith("GB")
    assert format_bytes(512 * 1024**2).endswith("MB")
    assert format_bytes(8 * 1024).endswith("KB")


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def test_plan_keeps_everything_resident_when_the_budget_is_ample():
    model = _TinyLM()
    plan = plan_residency(model, "1GB", bandwidth_gb_s=10.0)

    assert plan.is_fully_resident
    assert len(plan.resident_layers) == plan.num_layers
    assert plan.streamed_layers == []
    assert plan.staging_bytes == 0
    assert plan.bytes_streamed_per_token == 0
    assert plan.tokens_per_s_ceiling == float("inf")
    assert plan.vram_saved_bytes == 0


def test_plan_streams_the_tail_when_the_budget_is_tight():
    model = _TinyLM(num_layers=8)
    layer_bytes = max(module_bytes(layer) for layer in model.model.layers)
    anchors = module_bytes(model) - sum(module_bytes(l) for l in model.model.layers)

    # Room for anchors, two staging slots and about three resident layers.
    budget = anchors + 5 * layer_bytes
    plan = plan_residency(
        model, budget, activation_reserve=0, max_context_tokens=0, bandwidth_gb_s=10.0
    )

    assert not plan.is_fully_resident
    assert plan.resident_layers + plan.streamed_layers == list(range(plan.num_layers))
    assert plan.streamed_layers, "a tight budget must stream something"
    # The plan must fit inside what it was given.
    assert plan.vram_weight_bytes <= plan.vram_budget_bytes
    # Streaming is contiguous and at the tail, so the prefetcher can run ahead.
    assert plan.streamed_layers[0] == len(plan.resident_layers)
    assert plan.bytes_streamed_per_token == len(plan.streamed_layers) * plan.layer_bytes


def test_plan_never_promises_more_than_the_bus_allows():
    model = _TinyLM(num_layers=8)
    layer_bytes = max(module_bytes(layer) for layer in model.model.layers)
    anchors = module_bytes(model) - sum(module_bytes(l) for l in model.model.layers)

    plan = plan_residency(
        model,
        anchors + 3 * layer_bytes,
        activation_reserve=0,
        max_context_tokens=0,
        bandwidth_gb_s=8.0,
    )
    expected_seconds = plan.bytes_streamed_per_token / (8.0 * 1024**3)

    assert plan.pcie_seconds_per_token == pytest.approx(expected_seconds)
    assert plan.tokens_per_s_ceiling == pytest.approx(1.0 / expected_seconds)


def test_plan_refuses_a_budget_that_cannot_hold_the_anchors():
    model = _TinyLM()
    with pytest.raises(InsufficientVRAMError, match="must stay resident"):
        plan_residency(model, "1kb", bandwidth_gb_s=10.0)


def test_kv_cache_estimate_respects_grouped_query_attention():
    full = SimpleNamespace(
        num_hidden_layers=4, num_attention_heads=8, hidden_size=64, num_key_value_heads=8
    )
    grouped = SimpleNamespace(
        num_hidden_layers=4, num_attention_heads=8, hidden_size=64, num_key_value_heads=2
    )

    assert estimate_kv_cache_bytes(grouped, 1024) == estimate_kv_cache_bytes(full, 1024) // 4


def test_kv_cache_estimate_returns_zero_for_an_opaque_config():
    assert estimate_kv_cache_bytes(SimpleNamespace(), 1024) == 0


# ---------------------------------------------------------------------------
# The runtime: losslessness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("resident_count", [0, 2, 5])
def test_streaming_output_is_identical(resident_count):
    """
    The whole point: moving weights in and out of staging slots must not change
    a single number. Run on CPU so this holds everyone to the invariant.
    """
    torch.manual_seed(0)
    model = _TinyLM(num_layers=6)
    model.eval()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 7))

    with torch.no_grad():
        expected = model(input_ids)

    plan = _plan_for(model, resident_count)
    with torch.no_grad():
        with StreamingModel(model, plan, device=torch.device("cpu"), pin_memory=False):
            actual = model(input_ids)

    assert torch.equal(expected, actual), "streamed execution changed the output"


def test_streaming_is_identical_across_repeated_passes():
    """Slot rotation must stay correct over many passes, not just the first."""
    torch.manual_seed(1)
    model = _TinyLM(num_layers=6)
    model.eval()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 4))

    with torch.no_grad():
        expected = model(input_ids)

    plan = _plan_for(model, resident_count=1)
    with torch.no_grad():
        with StreamingModel(model, plan, device=torch.device("cpu"), pin_memory=False):
            for pass_index in range(5):
                assert torch.equal(model(input_ids), expected), f"pass {pass_index} diverged"


def test_streaming_restores_the_model_on_exit():
    torch.manual_seed(2)
    model = _TinyLM(num_layers=4)
    model.eval()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 3))

    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    plan = _plan_for(model, resident_count=1)

    with StreamingModel(model, plan, device=torch.device("cpu"), pin_memory=False):
        pass

    for name, param in model.named_parameters():
        assert torch.equal(param, before[name]), f"{name} was altered"

    with torch.no_grad():
        model(input_ids)  # still runnable without the runtime installed


def test_streaming_counts_what_it_moved():
    model = _TinyLM(num_layers=6)
    model.eval()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 3))

    plan = _plan_for(model, resident_count=2)
    with torch.no_grad():
        with StreamingModel(model, plan, device=torch.device("cpu"), pin_memory=False) as streaming:
            model(input_ids)
            model(input_ids)
            stats = streaming.stats()

    assert stats["forward_passes"] == 2
    assert stats["streamed_layers"] == len(plan.streamed_layers)
    # Each streamed layer is fetched once per pass.
    assert stats["bytes_transferred"] == 2 * len(plan.streamed_layers) * plan.layer_bytes


# ---------------------------------------------------------------------------
# Page-locked host memory (CUDA only)
# ---------------------------------------------------------------------------

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _streamed_host_tensors(streaming):
    return [t for params in streaming._host_params.values() for t in params.values()]


@needs_cuda
@pytest.mark.parametrize("pin_reserve", ["0B", "1024GB"])
def test_streaming_on_cuda_is_identical_pinned_or_not(pin_reserve):
    """Pinned (async) and pageable (fallback) host copies must both be lossless."""
    torch.manual_seed(3)
    device = torch.device("cuda")
    model = _TinyLM(num_layers=6).to(device)
    model.eval()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 5), device=device)

    with torch.no_grad():
        expected = model(input_ids)

    plan = _plan_for(model, resident_count=1)
    with torch.no_grad():
        with StreamingModel(model, plan, device=device, pin_reserve=pin_reserve) as streaming:
            pinned = [t.is_pinned() for t in _streamed_host_tensors(streaming)]
            for pass_index in range(3):
                assert torch.equal(model(input_ids), expected), f"pass {pass_index} diverged"

    if pin_reserve == "0B":
        assert all(pinned), "streamed layers should be page-locked"
    else:
        assert not any(pinned), "a reserve larger than RAM must leave layers pageable"


@needs_cuda
def test_pinning_costs_exactly_the_streamed_bytes_and_is_released():
    """No power-of-two rounding while streaming; no locked RAM left afterwards."""
    device = torch.device("cuda")
    model = _TinyLM(num_layers=6).to(device)
    plan = _plan_for(model, resident_count=2)

    with StreamingModel(model, plan, device=device, pin_reserve="0B") as streaming:
        host = _streamed_host_tensors(streaming)
        payload = sum(t.numel() * t.element_size() for t in host)
        locked = sum(flat.numel() for flat in streaming._registered)
        # Only per-tensor 512-byte alignment padding is allowed on top.
        assert payload <= locked <= payload + 512 * len(host)

    assert streaming._registered == []
    assert not any(t.is_pinned() for t in host)


# ---------------------------------------------------------------------------
# Fast mode: computing layers on the CPU
# ---------------------------------------------------------------------------

def _sizes(model):
    layer_bytes = max(module_bytes(layer) for layer in model.model.layers)
    anchors = module_bytes(model) - sum(module_bytes(l) for l in model.model.layers)
    return layer_bytes, anchors


def _fast_plan(model, budget, cpu_layer_ms, bandwidth_gb_s=10.0, max_context_tokens=0):
    return plan_residency(
        model,
        budget,
        mode="fast",
        activation_reserve=0,
        max_context_tokens=max_context_tokens,
        bandwidth_gb_s=bandwidth_gb_s,
        cpu_layer_ms=cpu_layer_ms,
        gpu_layer_ms=0.01,
    )


def test_fast_plan_computes_on_the_cpu_when_that_beats_the_bus():
    model = _TinyLM(num_layers=8)
    layer_bytes, anchors = _sizes(model)

    # A near-zero bus makes every transfer slower than any CPU layer.
    plan = _fast_plan(model, anchors + 3 * layer_bytes, cpu_layer_ms=1.0, bandwidth_gb_s=1e-6)

    assert plan.cpu_layers and not plan.streamed_layers
    # One slot is kept for prompts; the rest of the budget is resident layers.
    assert plan.staging_slots == 1
    assert len(plan.resident_layers) == 2
    assert plan.resident_layers + plan.cpu_layers == list(range(plan.num_layers))
    assert not plan.is_exact
    assert plan.vram_weight_bytes <= plan.vram_budget_bytes


def test_fast_plan_streams_when_the_bus_beats_the_cpu():
    model = _TinyLM(num_layers=8)
    layer_bytes, anchors = _sizes(model)

    plan = _fast_plan(model, anchors + 5 * layer_bytes, cpu_layer_ms=1e6)

    assert plan.streamed_layers and not plan.cpu_layers
    assert plan.is_exact


def test_fast_plan_never_mixes_cpu_and_streamed_layers():
    """Mixing was measured to be slower than either pure plan (see _plan_fast)."""
    model = _TinyLM(num_layers=8)
    layer_bytes, anchors = _sizes(model)

    for cpu_layer_ms in (1e-6, 1e-3, 1.0, 1e6):
        for extra in range(1, 9):
            plan = _fast_plan(model, anchors + extra * layer_bytes, cpu_layer_ms=cpu_layer_ms)
            assert not (plan.cpu_layers and plan.streamed_layers)


def test_fast_plan_reserves_kv_cache_only_for_gpu_layers():
    model = _TinyLM(num_layers=8)
    layer_bytes, anchors = _sizes(model)
    full_kv = estimate_kv_cache_bytes(model.config, 64, 4)

    plan = _fast_plan(
        model, anchors + full_kv + 3 * layer_bytes, cpu_layer_ms=1.0,
        bandwidth_gb_s=1e-6, max_context_tokens=64,
    )

    gpu_layers = len(plan.resident_layers)
    assert plan.cpu_layers
    assert plan.kv_cache_reserve_bytes == pytest.approx(full_kv * gpu_layers / 8, abs=1)


def test_fast_plan_fits_below_the_exact_floor():
    """With no staging slots, fast mode runs on budgets exact mode must refuse."""
    model = _TinyLM(num_layers=8)
    layer_bytes, anchors = _sizes(model)
    budget = anchors + layer_bytes // 2  # anchors fit, not even one slot does

    with pytest.raises(InsufficientVRAMError):
        plan_residency(model, budget, activation_reserve=0, max_context_tokens=0, bandwidth_gb_s=10.0)

    plan = _fast_plan(model, budget, cpu_layer_ms=1.0)
    assert plan.resident_layers == [] and len(plan.cpu_layers) == plan.num_layers
    # No room for the prompt slot either: prompts fall back to the CPU.
    assert plan.staging_slots == 0
    assert plan.vram_weight_bytes <= budget


def test_plan_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        plan_residency(_TinyLM(), "1GB", bandwidth_gb_s=10.0, mode="fastest")


def _cpu_plan(model, resident_count, prompt_slot=False):
    plan = _plan_for(model, resident_count)
    plan.cpu_layers, plan.streamed_layers = plan.streamed_layers, []
    plan.staging_slots = 1 if prompt_slot else 0
    plan.mode = "fast"
    return plan


@pytest.mark.parametrize("resident_count", [0, 3])
def test_cpu_layers_run_and_restore_on_a_cpu_device(resident_count):
    torch.manual_seed(4)
    model = _TinyLM(num_layers=6)
    model.eval()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 4))
    with torch.no_grad():
        expected = model(input_ids)

    with torch.no_grad():
        with StreamingModel(model, _cpu_plan(model, resident_count), device=torch.device("cpu")):
            assert torch.equal(model(input_ids), expected)
        assert torch.equal(model(input_ids), expected)


@needs_cuda
@pytest.mark.parametrize("resident_count", [0, 2, 5])
def test_cpu_layers_on_cuda_match_the_gpu_closely(resident_count):
    """Not bit-identical by design, but the same function: fp32 agrees to rounding."""
    torch.manual_seed(5)
    device = torch.device("cuda")
    model = _TinyLM(num_layers=6).to(device)
    model.eval()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 5), device=device)
    with torch.no_grad():
        expected = model(input_ids)

    plan = _cpu_plan(model, resident_count)
    with torch.no_grad():
        with StreamingModel(model, plan, device=device):
            for layer_idx in plan.cpu_layers:
                assert next(model.model.layers[layer_idx].parameters()).device.type == "cpu"
            actual = model(input_ids)

    assert actual.device.type == "cuda", "output must come back to the device"
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_fast_plan_keeps_exactness_unless_the_cpu_is_clearly_faster():
    """Identical output is worth something: a marginal CPU win must not buy it away."""
    from paramap.residency import _CPU_IN_DECODE_FACTOR, _FAST_MIN_SPEEDUP

    model = _TinyLM(num_layers=8)
    layer_bytes, anchors = _sizes(model)
    budget = anchors + 3 * layer_bytes  # cpu: 2 resident + 6 CPU + prompt slot; stream: 1 + 7
    bandwidth = 1e-3
    transfer_ms = layer_bytes / (bandwidth * 1024**3) * 1e3
    stream_ms = 7 * transfer_ms

    def plan_with_cpu_speedup(speedup):
        cpu_ms_in_decode = stream_ms / speedup / 6
        return _fast_plan(
            model, budget, cpu_layer_ms=cpu_ms_in_decode / _CPU_IN_DECODE_FACTOR,
            bandwidth_gb_s=bandwidth,
        )

    assert plan_with_cpu_speedup(_FAST_MIN_SPEEDUP * 0.95).is_exact
    assert not plan_with_cpu_speedup(_FAST_MIN_SPEEDUP * 1.5).is_exact


def test_fast_plan_counts_the_prompt_slot_against_the_budget():
    model = _TinyLM(num_layers=8)
    layer_bytes, anchors = _sizes(model)

    for extra in range(1, 8):
        plan = _fast_plan(model, anchors + extra * layer_bytes, cpu_layer_ms=1.0, bandwidth_gb_s=1e-6)
        assert plan.cpu_layers
        assert plan.staging_bytes == plan.staging_slots * layer_bytes
        assert plan.vram_weight_bytes <= plan.vram_budget_bytes


# ---------------------------------------------------------------------------
# Prompts through the slot, decode on the CPU, on a real (tiny) Qwen2
# ---------------------------------------------------------------------------

def _tiny_qwen2(device):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(6)
    config = Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=6,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
    )
    return Qwen2ForCausalLM(config).to(device).eval()


@torch.no_grad()
def _prompt_then_decode(model, prompt, steps, device):
    from transformers import DynamicCache

    cache = DynamicCache()
    logits = [model(prompt.to(device), past_key_values=cache, use_cache=True).logits[0, -1]]
    token = logits[-1].argmax().view(1, 1)
    for _ in range(steps):
        logits.append(model(token.to(device), past_key_values=cache, use_cache=True).logits[0, -1])
        token = logits[-1].argmax().view(1, 1)
    return torch.stack(logits), cache


@needs_cuda
@pytest.mark.parametrize("prompt_slot", [True, False])
def test_cpu_layers_handle_prompt_then_decode(prompt_slot):
    """With the slot, prompts run CPU layers on the GPU and hand the KV cache to the CPU."""
    device = torch.device("cuda")
    model = _tiny_qwen2(device)
    prompt = torch.randint(0, 64, (1, 12))
    expected, _ = _prompt_then_decode(model, prompt, steps=6, device=device)

    plan = _cpu_plan(model, resident_count=2, prompt_slot=prompt_slot)
    with StreamingModel(model, plan, device=device) as streaming:
        actual, cache = _prompt_then_decode(model, prompt, steps=6, device=device)
        stats = streaming.stats()
        for layer_idx in plan.cpu_layers:
            assert next(model.model.layers[layer_idx].parameters()).device.type == "cpu"
            assert cache.layers[layer_idx].keys.device.type == "cpu"
        for layer_idx in plan.resident_layers:
            assert cache.layers[layer_idx].keys.device.type == "cuda"

    # One prompt pass: each CPU layer borrows the slot once, decode never does.
    assert stats["prefill_layer_runs"] == (len(plan.cpu_layers) if prompt_slot else 0)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

