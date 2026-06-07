from pathlib import Path

import pytest
import torch

from vllm.model_executor.models.deepseek_v4_flash.attention import (
    factorized_attention_projection_reference,
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
        assert layer.attention_query_b is not None
        assert layer.attention_key_value is not None
        assert layer.attention_output_a is not None
        assert layer.attention_output_b is not None

        assert layer.attention_query_a.dims == (4096, 1024)
        assert layer.attention_query_b.dims == (1024, 32768)
        # The target exposes a 512-wide combined attn_kv tensor. This task
        # records the shape only; the key/value semantic split is not derived
        # here and must be resolved before real attention execution.
        assert layer.attention_key_value.dims == (4096, 512)
        assert layer.attention_output_a.dims == (4096, 8192)
        assert layer.attention_output_b.dims == (8192, 4096)


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
