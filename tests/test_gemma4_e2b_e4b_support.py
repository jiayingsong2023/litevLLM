import json
import os
from types import SimpleNamespace

import pytest
from transformers import AutoTokenizer

from vllm.model_executor.models.lite_config import LiteConfig


@pytest.fixture
def e2b_text_config():
    return SimpleNamespace(
        **json.load(open("models/gemma-4-E2B-it-AWQ-INT4/config.json"))["text_config"]
    )


@pytest.fixture
def e4b_text_config():
    return SimpleNamespace(
        **json.load(open("models/gemma-4-E4B-it/config.json"))["text_config"]
    )


def test_lite_config_e2b_helpers(e2b_text_config):
    cfg = LiteConfig(e2b_text_config)
    assert cfg.use_double_wide_mlp is True
    assert cfg.num_kv_shared_layers == 20
    assert cfg.hidden_size_per_layer_input == 256
    assert cfg.vocab_size_per_layer_input == 262144
    assert cfg.ple_enabled() is True
    assert cfg.is_kv_shared_layer(14) is False
    assert cfg.is_kv_shared_layer(15) is True
    assert cfg.is_kv_shared_layer(34) is True
    assert cfg.effective_intermediate_size(0) == 6144
    assert cfg.effective_intermediate_size(15) == 12288


def test_lite_config_e4b_helpers(e4b_text_config):
    cfg = LiteConfig(e4b_text_config)
    assert cfg.use_double_wide_mlp is False
    assert cfg.num_kv_shared_layers == 18
    assert cfg.hidden_size_per_layer_input == 256
    assert cfg.ple_enabled() is True
    assert cfg.is_kv_shared_layer(23) is False
    assert cfg.is_kv_shared_layer(24) is True
    assert cfg.effective_intermediate_size(24) == 10240


def test_mlp_double_wide_shape(e2b_text_config):
    import torch

    from vllm.model_executor.models.gemma4.mlp import Gemma4MLP

    cfg = LiteConfig(e2b_text_config)
    mlp0 = Gemma4MLP(cfg, None, "layers.0", layer_idx=0)
    mlp15 = Gemma4MLP(cfg, None, "layers.15", layer_idx=15)
    assert mlp0.gate_proj.output_size == 6144
    assert mlp0.down_proj.input_size == 6144
    assert mlp15.gate_proj.output_size == 12288
    assert mlp15.down_proj.input_size == 12288

    # Forward shape sanity with dummy dense weights
    for mlp in (mlp0, mlp15):
        mlp.gate_proj.weight = torch.nn.Parameter(
            torch.zeros(
                mlp.gate_proj.output_size, cfg.hidden_size, dtype=torch.float16
            ),
            requires_grad=False,
        )
        mlp.up_proj.weight = torch.nn.Parameter(
            torch.zeros(mlp.up_proj.output_size, cfg.hidden_size, dtype=torch.float16),
            requires_grad=False,
        )
        mlp.down_proj.weight = torch.nn.Parameter(
            torch.zeros(cfg.hidden_size, mlp.down_proj.input_size, dtype=torch.float16),
            requires_grad=False,
        )
        x = torch.randn(1, 12, cfg.hidden_size, dtype=torch.float16)
        out = mlp(x)
        assert out.shape == (1, 12, cfg.hidden_size)


def test_kv_shared_donor_selection(e2b_text_config):
    from vllm.model_executor.models.gemma4.model import Gemma4TextModel

    # Need a minimal quant_config; None works for shape tests because layers
    # don't build weights until load.
    model = Gemma4TextModel(e2b_text_config, None)
    # Shared layers do not own k/v projection or normalization; they reuse the
    # donor layer's KV cache state via kv_scale_cache_idx (HF Gemma4's
    # shared_kv_states semantics).
    assert model.layers[15].self_attn.is_kv_shared_layer is True
    assert model.layers[15].self_attn.k_proj is None
    assert model.layers[15].self_attn.v_proj is None
    assert model.layers[15].self_attn.k_norm is None
    # Layer 15 is sliding_attention; donor is the last non-shared sliding layer 13.
    assert model.layers[15].self_attn.kv_scale_cache_idx == 13
    # Layer 19 is full_attention; donor is the last non-shared full layer 14.
    assert model.layers[19].self_attn.is_kv_shared_layer is True
    assert model.layers[19].self_attn.kv_scale_cache_idx == 14
    # Non-shared layer points at itself and owns k/v modules.
    assert model.layers[10].self_attn.is_kv_shared_layer is False
    assert model.layers[10].self_attn.kv_scale_cache_idx == 10
    assert model.layers[10].self_attn.k_proj is not None
    assert model.layers[10].self_attn.v_proj is not None


def test_ple_shape(e2b_text_config):
    import torch

    from vllm.model_executor.models.gemma4.model import Gemma4TextModel

    cfg = LiteConfig(e2b_text_config)
    model = Gemma4TextModel(e2b_text_config, None)
    assert model.embed_tokens_per_layer is not None
    assert model.per_layer_model_projection is not None
    assert model.per_layer_projection_norm is not None

    B, S = 1, 5
    input_ids = torch.randint(0, cfg.vocab_size, (B, S), dtype=torch.long)
    inputs_embeds = torch.randn(B, S, cfg.hidden_size, dtype=torch.float16)
    ple = model._compute_per_layer_inputs(input_ids, inputs_embeds)
    assert ple.shape == (B, S, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input)


def test_e2b_asymmetric_packed_int4_dequant_matches_reference():
    """
    Gemma4-E2B AWQ-INT4 uses compressed-tensors pack-quantized asymmetric weights:
    weight_packed [out, in//8], weight_scale [out, in//group_size],
    weight_zero_point [out//8, in//group_size].  Verify our fallback dequant
    matches the compressed_tensors reference implementation bit-for-bit.
    """
    import torch
    from compressed_tensors.compressors.quantized_compressors.pack_quantized import (
        pack_to_int32,
        unpack_from_int32,
    )
    from compressed_tensors.quantization.lifecycle.forward import dequantize

    from vllm.model_executor.layers.quantization.tensor import dequantize_awq_pytorch

    out_features, in_features, group_size = 64, 128, 32
    torch.manual_seed(42)

    # Build per-group scale/zero_point and signed quantized weights manually so we
    # control the values without depending on compressed-tensors quantize internals.
    dense_weight = torch.randn(out_features, in_features, dtype=torch.float32)
    n_groups = in_features // group_size
    scale = torch.empty(out_features, n_groups, dtype=torch.float32)
    zero_point_unsigned = torch.empty(out_features, n_groups, dtype=torch.int32)
    qweight_signed = torch.empty(out_features, in_features, dtype=torch.int8)
    for g in range(n_groups):
        col_start = g * group_size
        col_end = col_start + group_size
        w_group = dense_weight[:, col_start:col_end]
        w_min = w_group.min(dim=1, keepdim=True).values
        w_max = w_group.max(dim=1, keepdim=True).values
        s = (w_max - w_min) / 15.0
        s[s == 0] = 1.0
        zp = (-w_min / s).round().to(torch.int32).clamp(0, 15)
        q = (w_group / s).round().to(torch.int32) + zp
        q = q.clamp(0, 15).to(torch.int8) - 8  # signed int8
        scale[:, g] = s.squeeze(1)
        zero_point_unsigned[:, g] = zp.squeeze(1)
        qweight_signed[:, col_start:col_end] = q

    weight_packed = pack_to_int32(qweight_signed, num_bits=4, packed_dim=1)
    packed_zp = pack_to_int32(
        zero_point_unsigned.to(torch.int8) - 8, num_bits=4, packed_dim=0
    )

    result = dequantize_awq_pytorch(
        weight_packed,
        scale,
        packed_zp,
        group_size=group_size,
    )

    ref_unpacked = unpack_from_int32(weight_packed, 4, (out_features, in_features))
    original_zp_shape = (out_features, in_features // group_size)
    ref_zp = unpack_from_int32(packed_zp, 4, original_zp_shape, packed_dim=0)
    reference = dequantize(ref_unpacked, scale, ref_zp, dtype=torch.float16)

    max_diff = (result.to(torch.float32) - reference.to(torch.float32)).abs().max()
    assert max_diff == 0.0, f"asymmetric dequant mismatch: {max_diff.item()}"


@pytest.mark.skipif(
    os.environ.get("RUN_GEMMA4_E2B_SMOKE") != "1",
    reason="Set RUN_GEMMA4_E2B_SMOKE=1 to load the E2B model",
)
def test_e2b_awq_q4_generates():
    import os

    os.environ["FASTINFERENCE_GEMMA4_ALLOW_INT4_KV"] = "1"
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained("models/gemma-4-E2B-it-AWQ-INT4")
    messages = [
        {"role": "user", "content": "The capital of France is"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    llm = LLM(
        model="models/gemma-4-E2B-it-AWQ-INT4",
        max_model_len=256,
        gpu_memory_utilization=0.85,
        max_num_batched_tokens=512,
    )
    outputs = llm.generate(prompt, max_tokens=16)
    assert outputs
    text = outputs[0].outputs[0].text
    assert isinstance(text, str) and len(text) > 0
    # Guard against the previous gibberish regression (e.g. " None,...").
    assert "Paris" in text, f"E2B output was incoherent: {text!r}"
    print("E2B output:", text)
    llm.shutdown()


def test_e2b_asymmetric_matmul_dispatches_to_gemv():
    import torch

    from tests.tools.gemma4_e2b_quant_helpers import make_asymmetric_packed_int4
    from vllm.model_executor.layers.quantization.tensor import (
        AWQWeight,
        dequantize_asymmetric_packed_int4_pytorch,
        get_awq_runtime_stats,
        reset_awq_runtime_stats,
    )

    qweight, scales, qzeros = make_asymmetric_packed_int4(2048, 1536)
    w = AWQWeight(qweight, scales, qzeros, group_size=32, prefix="test")
    x = torch.randn(1, 1, 1536, dtype=torch.float16, device="cuda")

    class FakeConfig:
        kernel_policy = {"awq_asymmetric_gemv": True}

    reset_awq_runtime_stats()
    out = w.matmul(x, config=FakeConfig())

    dense = dequantize_asymmetric_packed_int4_pytorch(qweight, scales, qzeros, 32)
    ref = torch.nn.functional.linear(x, dense)
    torch.testing.assert_close(out, ref, atol=0.02, rtol=0.02)

    stats = get_awq_runtime_stats()
    assert stats.get("awq_asymmetric_gemv_success", 0) >= 1, stats
