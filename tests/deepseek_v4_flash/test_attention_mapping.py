from pathlib import Path

import pytest
import torch

from vllm.model_executor.models.deepseek_v4_flash.attention import (
    factorized_attention_projection_reference,
    latent_kv_projection_reference,
    q_lora_attention_projection_reference,
    split_combined_kv_reference,
)
from vllm.model_executor.models.deepseek_v4_flash.weight_store import (
    open_deepseek_v4_flash_weight_store,
)

TARGET_GGUF = Path(
    "models/DeepSeek-V4-Flash-ds4/"
    "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
)


@pytest.mark.skipif(not TARGET_GGUF.exists(), reason="target DeepSeek V4 GGUF absent")
def test_real_attention_tensor_shape_mapping_for_layer_0() -> None:
    with open_deepseek_v4_flash_weight_store(TARGET_GGUF) as store:
        layer = store.bindings.layers[0]

        assert layer.attention_query_a is not None
        assert layer.attention_query_a_norm is not None
        assert layer.attention_query_b is not None
        assert layer.attention_key_value is not None
        assert layer.attention_key_value_a_norm is not None
        assert layer.attention_output_a is not None
        assert layer.attention_output_b is not None
        assert layer.attention_sinks is not None

        assert layer.attention_query_a.dims == (4096, 1024)
        assert layer.attention_query_a_norm.dims == (1024,)
        assert layer.attention_query_b.dims == (1024, 32768)
        # DeepSeek V4 Flash uses a single 512-wide latent for both K and V.
        assert layer.attention_key_value.dims == (4096, 512)
        assert layer.attention_key_value_a_norm.dims == (512,)
        assert layer.attention_output_a.dims == (4096, 8192)
        assert layer.attention_output_b.dims == (8192, 4096)
        assert layer.attention_sinks.dims == (64,)


@pytest.mark.skipif(not TARGET_GGUF.exists(), reason="target DeepSeek V4 GGUF absent")
def test_real_layer2_compressor_and_indexer_shape_mapping() -> None:
    with open_deepseek_v4_flash_weight_store(TARGET_GGUF) as store:
        layer = store.bindings.layers[2]
        assert layer.attention_compressor is not None
        assert layer.indexer is not None

        assert layer.attention_compressor.kv.dims == (4096, 1024)
        assert layer.attention_compressor.gate.dims == (4096, 1024)
        assert layer.attention_compressor.ape.dims == (1024, 4)
        assert layer.attention_compressor.norm.dims == (512,)

        assert layer.indexer.query_b.dims == (1024, 8192)
        assert layer.indexer.projection.dims == (4096, 64)
        assert layer.indexer.compressor.kv.dims == (4096, 256)
        assert layer.indexer.compressor.gate.dims == (4096, 256)
        assert layer.indexer.compressor.ape.dims == (256, 4)
        assert layer.indexer.compressor.norm.dims == (128,)

        # Decode one representative matrix from each path to prove the real
        # layer-2 tensors are accessible through the typed matrix accessor.
        assert store.decode_matrix(layer.attention_compressor.kv).shape == (
            1024,
            4096,
        )
        assert store.decode_matrix(layer.indexer.projection).shape == (64, 4096)


def test_factorized_attention_projection_reference_returns_b_output_shape() -> None:
    out = factorized_attention_projection_reference(
        torch.ones(4),
        torch.eye(4),
        torch.ones((4, 8)),
    )

    assert out.shape == (8,)


def test_factorized_attention_projection_reference_uses_gguf_orientation() -> None:
    hidden = torch.arange(1.0, 5.0)
    q_a = torch.tensor(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    q_b = torch.ones((3, 8))

    out = factorized_attention_projection_reference(hidden, q_a, q_b)

    expected = q_b.transpose(0, 1).matmul(q_a.transpose(0, 1).matmul(hidden))
    assert out.shape == (8,)
    torch.testing.assert_close(out, expected)


def test_q_lora_attention_projection_reference_applies_norm_between_factors() -> None:
    hidden = torch.tensor([3.0, 4.0])
    q_a = torch.eye(2)
    q_norm = torch.tensor([2.0, 1.0])
    q_b = torch.ones((2, 3))

    out = q_lora_attention_projection_reference(hidden, q_a, q_norm, q_b)

    q_latent = q_a.transpose(0, 1).matmul(hidden)
    q_latent = q_latent * torch.rsqrt(q_latent.pow(2).mean() + 1e-6) * q_norm
    expected = q_b.transpose(0, 1).matmul(q_latent)
    torch.testing.assert_close(out, expected)


def test_latent_kv_projection_reference_returns_single_kv_latent() -> None:
    hidden = torch.tensor([1.0, 2.0])
    kv_weight = torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    kv_norm = torch.ones(3)

    out = latent_kv_projection_reference(hidden, kv_weight, kv_norm)

    expected = kv_weight.transpose(0, 1).matmul(hidden)
    expected = expected * torch.rsqrt(expected.pow(2).mean() + 1e-6)
    assert out.shape == (3,)
    torch.testing.assert_close(out, expected)


def test_split_combined_kv_reference_returns_key_and_value_views() -> None:
    kv = torch.arange(6)

    key, value = split_combined_kv_reference(kv, key_width=2, value_width=4)

    torch.testing.assert_close(key, torch.tensor([0, 1]))
    torch.testing.assert_close(value, torch.tensor([2, 3, 4, 5]))


@pytest.mark.parametrize(
    ("kv", "key_width", "value_width", "message"),
    [
        (torch.ones((1, 4)), 2, 2, "kv must be 1-D"),
        (torch.ones(4), -1, 5, "widths must be positive"),
        (torch.ones(4), 2, -1, "widths must be positive"),
        (torch.ones(4), 0, 4, "widths must be positive"),
        (torch.ones(4), 4, 0, "widths must be positive"),
        (torch.ones(4), 2, 3, "kv length must equal key_width \\+ value_width"),
    ],
)
def test_split_combined_kv_reference_validates_shape_and_width(
    kv: torch.Tensor,
    key_width: int,
    value_width: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        split_combined_kv_reference(
            kv,
            key_width=key_width,
            value_width=value_width,
        )
