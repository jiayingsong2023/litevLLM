import pytest
import torch

from vllm.model_executor.models.deepseek_v4_flash.attention import (
    apply_rope_to_tail_reference,
    grouped_output_projection_reference,
    per_head_rms_norm_reference,
    raw_swa_attention_reference,
    shared_kv_swa_attention_reference,
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


def test_per_head_rms_norm_reference_normalizes_each_head() -> None:
    query = torch.tensor([[3.0, 4.0], [1.0, 0.0]])

    out = per_head_rms_norm_reference(query, eps=0.0)

    expected = query * torch.rsqrt(query.pow(2).mean(dim=-1, keepdim=True))
    torch.testing.assert_close(out, expected)


def test_apply_rope_to_tail_reference_rotates_only_tail() -> None:
    vector = torch.tensor([[9.0, 8.0, 1.0, 0.0]])

    out = apply_rope_to_tail_reference(
        vector,
        token_idx=1,
        rotary_dim=2,
        theta=10000.0,
    )

    assert out[0, 0].item() == 9.0
    assert out[0, 1].item() == 8.0
    torch.testing.assert_close(
        out[0, 2:],
        torch.tensor([torch.cos(torch.tensor(1.0)), torch.sin(torch.tensor(1.0))]),
    )


def test_shared_kv_swa_attention_reference_uses_sink_denominator() -> None:
    queries = torch.tensor([[2.0, 0.0]])
    kv_rows = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    sinks = torch.tensor([0.5])

    out = shared_kv_swa_attention_reference(
        queries,
        kv_rows,
        sinks,
        softmax_scale=1.0,
    )

    logits = torch.tensor([4.0, 0.0, 0.5])
    probs = torch.softmax(logits, dim=0)
    expected = probs[:2].matmul(kv_rows)
    torch.testing.assert_close(out[0], expected)


def test_grouped_output_projection_reference_matches_manual_grouping() -> None:
    attention_output = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )
    output_a = torch.zeros((4, 4))
    output_a[:, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    output_a[:, 1] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    output_a[:, 2] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    output_a[:, 3] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    output_b = torch.eye(4)

    out = grouped_output_projection_reference(
        attention_output,
        output_a,
        output_b,
        output_groups=2,
    )

    torch.testing.assert_close(out, torch.tensor([1.0, 2.0, 5.0, 6.0]))
