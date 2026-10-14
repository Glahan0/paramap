"""
Tests for loading weights as views into memory-mapped safetensors files.

What must hold: the mapped model computes exactly what the ordinarily-loaded
one does, its weights are the file's bytes rather than copies, and running a
plan over it leaves it pointing at the file again afterwards.
"""

import pytest
import torch

from paramap.loader import SafetensorsMmap, is_file_backed, load_model_mmap, safetensors_files
from paramap.residency import ResidencyPlan, StreamingModel, module_bytes

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _save_tiny_qwen2(directory, tie_word_embeddings=False, dtype=torch.float32):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(11)
    config = Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=6,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        tie_word_embeddings=tie_word_embeddings,
    )
    model = Qwen2ForCausalLM(config).to(dtype).eval()
    model.save_pretrained(directory, max_shard_size="20KB")  # several shards
    return model


def _mapping_ranges(store):
    ranges = []
    for mapping in store._maps:
        start = torch.frombuffer(mapping, dtype=torch.uint8, count=1).data_ptr()
        ranges.append((start, start + len(mapping)))
    return ranges


@pytest.mark.parametrize("tied", [False, True])
def test_mapped_model_computes_exactly_what_the_original_does(tmp_path, tied):
    original = _save_tiny_qwen2(tmp_path, tie_word_embeddings=tied)
    assert len(safetensors_files(str(tmp_path))) > 1

    model, _ = load_model_mmap(str(tmp_path), torch.float32)
    input_ids = torch.randint(0, 64, (1, 9))
    with torch.no_grad():
        assert torch.equal(model(input_ids).logits, original(input_ids).logits)

    assert is_file_backed(model)
    if tied:
        assert model.lm_head.weight is model.model.embed_tokens.weight


def test_mapped_weights_are_views_into_the_file(tmp_path):
    _save_tiny_qwen2(tmp_path)
    model, _ = load_model_mmap(str(tmp_path), torch.float32)

    ranges = _mapping_ranges(model._paramap_store)
    for name, param in model.named_parameters():
        ptr = param.data_ptr()
        assert any(lo <= ptr < hi for lo, hi in ranges), f"{name} was copied out of the mapping"


def test_a_dtype_mismatch_is_converted_not_mapped(tmp_path, capsys):
    original = _save_tiny_qwen2(tmp_path)
    model, _ = load_model_mmap(str(tmp_path), torch.bfloat16)

    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    assert "converted" in capsys.readouterr().out
    first = next(original.parameters())
    assert torch.equal(next(model.parameters()), first.to(torch.bfloat16))


def test_the_store_reads_every_saved_tensor(tmp_path):
    original = _save_tiny_qwen2(tmp_path)
    store = SafetensorsMmap(safetensors_files(str(tmp_path)))
    for name, param in original.state_dict().items():
        assert torch.equal(store.tensor(name), param), name


def _exact_plan(model, resident_count):
    layers = model.model.layers
    layer_bytes = max(module_bytes(layer) for layer in layers)
    anchor_bytes = module_bytes(model) - sum(module_bytes(layer) for layer in layers)
    return ResidencyPlan(
        num_layers=len(layers),
        layer_bytes=layer_bytes,
        anchor_bytes=anchor_bytes,
        vram_budget_bytes=anchor_bytes + (resident_count + 2) * layer_bytes,
        kv_cache_reserve_bytes=0,
        activation_reserve_bytes=0,
        staging_slots=2,
        resident_layers=list(range(resident_count)),
        streamed_layers=list(range(resident_count, len(layers))),
        bandwidth_gb_s=10.0,
    )


@needs_cuda
@pytest.mark.parametrize("pin_reserve", ["0B", "1024GB"])
def test_streaming_from_the_file_is_identical_and_restores_the_views(tmp_path, pin_reserve):
    """Pinned (RAM tier) or straight from the mapping (disk tier): same output."""
    _save_tiny_qwen2(tmp_path)
    device = torch.device("cuda")
    model, _ = load_model_mmap(str(tmp_path), torch.float32)
    input_ids = torch.randint(0, 64, (1, 7), device=device)
    views = {name: param.data_ptr() for name, param in model.named_parameters()}

    baseline = _exact_plan(model, resident_count=6)
    baseline.streamed_layers, baseline.staging_slots = [], 0
    with torch.no_grad():
        with StreamingModel(model, baseline, device=device):
            expected = model(input_ids).logits

        for resident_count in (0, 3):
            plan = _exact_plan(model, resident_count)
            with StreamingModel(model, plan, device=device, pin_reserve=pin_reserve):
                for pass_index in range(2):
                    assert torch.equal(model(input_ids).logits, expected), pass_index

    # Every weight points back into the file, ready for the next plan.
    for name, param in model.named_parameters():
        assert param.device.type == "cpu"
        assert param.data_ptr() == views[name], f"{name} was left on a copy"


def _budget_for(model, resident_count):
    layers = model.model.layers
    layer_bytes = max(module_bytes(layer) for layer in layers)
    anchor_bytes = module_bytes(model) - sum(module_bytes(layer) for layer in layers)
    return anchor_bytes + (resident_count + 2) * layer_bytes, layer_bytes


def test_plan_warns_when_off_gpu_layers_exceed_free_ram(tmp_path):
    from paramap.residency import _HOST_RAM_RESERVE, plan_residency

    _save_tiny_qwen2(tmp_path)
    model, _ = load_model_mmap(str(tmp_path), torch.float32)
    budget, layer_bytes = _budget_for(model, resident_count=2)

    def plan_with_free(free_bytes):
        return plan_residency(
            model, budget, host_ram_available=free_bytes, activation_reserve=0,
            max_context_tokens=0, bandwidth_gb_s=10.0,
        )

    roomy = plan_with_free(_HOST_RAM_RESERVE + 100 * layer_bytes)
    assert roomy.host_overflow_bytes == 0 and roomy.ram_warning() == ""
    assert "RAM too small" not in roomy.describe()

    tight = plan_with_free(_HOST_RAM_RESERVE + layer_bytes)
    assert tight.host_overflow_bytes == tight.host_weight_bytes - layer_bytes
    assert "RAM too small" in tight.describe()


def test_no_ram_warning_for_a_model_already_in_ram(tmp_path):
    """Its weights are already resident, so free RAM says nothing."""
    from paramap.residency import plan_residency

    model = _save_tiny_qwen2(tmp_path)
    budget, _ = _budget_for(model, resident_count=2)
    plan = plan_residency(
        model, budget, host_ram_available=1, activation_reserve=0,
        max_context_tokens=0, bandwidth_gb_s=10.0,
    )
    assert plan.host_overflow_bytes == 0
