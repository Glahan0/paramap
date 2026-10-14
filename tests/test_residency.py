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
