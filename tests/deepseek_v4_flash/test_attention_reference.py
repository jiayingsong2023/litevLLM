import pytest
import torch

from vllm.model_executor.models.deepseek_v4_flash.attention import (
    raw_swa_attention_reference,
)


def test_raw_swa_attention_reference_scores_supplied_causal_window() -> None:
    query = torch.tensor([2.0, 0.0], dtype=torch.float16)
    keys = torch.tensor([[0.0, 1.0], [2.0, 0.0]], dtype=torch.bfloat16)
    values = torch.tensor([[10.0, 0.0], [0.0, 20.0]], dtype=torch.float16)

    out = raw_swa_attention_reference(query, keys, values)

    expected_scores = keys.to(torch.float32).matmul(query.to(torch.float32))
    expected_scores = expected_scores / torch.sqrt(torch.tensor(2.0))
    expected_probs = torch.softmax(expected_scores, dim=0)
    expected = expected_probs.matmul(values.to(torch.float32))
    torch.testing.assert_close(out, expected)


def test_raw_swa_attention_reference_returns_float32_value_vector() -> None:
    query = torch.tensor([1.0, -1.0], dtype=torch.float16)
    keys = torch.tensor([[1.0, 0.0]], dtype=torch.float16)
    values = torch.tensor([[3.0, 4.0, 5.0]], dtype=torch.bfloat16)

    out = raw_swa_attention_reference(query, keys, values)

    assert out.shape == (3, )
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, torch.tensor([3.0, 4.0, 5.0]))


@pytest.mark.parametrize(
    ("query", "keys", "values", "message"),
    [
        (
            torch.ones((1, 2)),
            torch.ones((1, 2)),
            torch.ones((1, 2)),
            "query must be 1-D",
        ),
        (
            torch.ones(2),
            torch.ones(2),
            torch.ones((1, 2)),
            "keys must be 2-D",
        ),
        (
            torch.ones(2),
            torch.ones((1, 2)),
            torch.ones(2),
            "values must be 2-D",
        ),
        (
            torch.ones(3),
            torch.ones((1, 2)),
            torch.ones((1, 2)),
            "key columns must match query size",
        ),
        (
            torch.ones(2),
            torch.ones((2, 2)),
            torch.ones((1, 2)),
            "keys and values must have the same row count",
        ),
    ],
)
def test_raw_swa_attention_reference_validates_shapes(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        raw_swa_attention_reference(query, keys, values)


def test_raw_swa_attention_reference_rejects_empty_keys_and_values() -> None:
    query = torch.ones(2)
    keys = torch.empty((0, 2))
    values = torch.empty((0, 3))

    with pytest.raises(ValueError, match="at least one key/value row"):
        raw_swa_attention_reference(query, keys, values)
