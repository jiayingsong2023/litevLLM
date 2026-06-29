import pytest
import torch

from vllm.model_executor.models.deepseek_v4_flash.attention import (
    apply_deepseek_layer_rope_to_tail_reference,
    apply_precomputed_rope_to_tail,
    apply_rope_to_tail_reference,
    build_deepseek_layer_rope_tables,
    compressor_pair_projection_reference,
    compressor_pool_state_reference,
    compressor_should_emit_reference,
    compressor_update_state_reference,
    grouped_output_projection_reference,
    indexer_query_projection_reference,
    indexer_scores_reference,
    indexer_topk_reference,
    indexer_weight_projection_reference,
    per_head_rms_norm_reference,
    raw_swa_attention_reference,
)
from vllm.model_executor.models.deepseek_v4_flash.attention import (
    shared_kv_swa_attention_reference as _shared_kv_swa_attention_reference,
)
from vllm.model_executor.models.deepseek_v4_flash.config import (
    DEEPSEEK_V4_FLASH_SHAPE,
)


def shared_kv_swa_attention_reference(
    queries: torch.Tensor,
    kv_rows: torch.Tensor,
    attn_sinks: torch.Tensor | None,
    *,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Wrapper that treats ``attn_sinks=None`` as no sink contribution.

    The underlying reference requires a sink tensor; a sink of ``-inf`` gives
    the same softmax denominator as omitting the sink entirely.
    """
    if attn_sinks is None:
        attn_sinks = torch.full(
            (queries.shape[0],),
            float("-inf"),
            dtype=queries.dtype,
            device=queries.device,
        )
    return _shared_kv_swa_attention_reference(
        queries,
        kv_rows,
        attn_sinks,
        softmax_scale=softmax_scale,
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

    assert out.shape == (3,)
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


def test_compressor_pair_projection_reference_projects_kv_and_gate() -> None:
    hidden = torch.tensor([1.0, 2.0])
    kv_weight = torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    gate_weight = torch.tensor([[0.0, 1.0, 1.0], [2.0, 0.0, 1.0]])

    kv, gate = compressor_pair_projection_reference(hidden, kv_weight, gate_weight)

    torch.testing.assert_close(kv, torch.tensor([1.0, 2.0, 4.0]))
    torch.testing.assert_close(gate, torch.tensor([4.0, 1.0, 3.0]))


def test_compressor_should_emit_reference_uses_absolute_position() -> None:
    assert compressor_should_emit_reference(token_idx=3, ratio=4) is True
    assert compressor_should_emit_reference(token_idx=2, ratio=4) is False


def test_indexer_query_projection_reference_returns_heads() -> None:
    qr_norm = torch.tensor([1.0, 2.0])
    weight = torch.ones((2, 6))

    out = indexer_query_projection_reference(
        qr_norm,
        weight,
        indexer_heads=3,
        indexer_head_dim=2,
    )

    assert out.shape == (3, 2)
    torch.testing.assert_close(out, torch.full((3, 2), 3.0))


def test_indexer_weight_projection_reference_returns_head_weights() -> None:
    out = indexer_weight_projection_reference(
        torch.tensor([1.0, 2.0]),
        torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]]),
    )

    torch.testing.assert_close(out, torch.tensor([1.0, 2.0, 4.0]))


def test_indexer_scores_and_topk_reference() -> None:
    query = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    weights = torch.tensor([1.0, 0.5])
    rows = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])

    scores = indexer_scores_reference(query, weights, rows, scale=1.0)
    topk = indexer_topk_reference(scores, top_k=2)

    torch.testing.assert_close(scores, torch.tensor([1.0, 2.0, 2.0]))
    torch.testing.assert_close(topk, torch.tensor([1, 2]))


def test_compressor_pool_state_reference_ratio4_uses_both_lanes() -> None:
    state_kv = torch.zeros((8, 4))
    state_score = torch.full((8, 4), -1000.0)
    state_kv[0, 0] = 1.0
    state_score[0, 0] = 0.0
    state_kv[4, 3] = 5.0
    state_score[4, 3] = 0.0

    pooled = compressor_pool_state_reference(
        state_kv,
        state_score,
        head_dim=2,
        ratio=4,
    )

    torch.testing.assert_close(pooled, torch.tensor([1.0, 5.0]))


def test_compressor_update_state_reference_emits_on_ratio_boundary() -> None:
    state_kv = torch.zeros((8, 4))
    state_score = torch.full((8, 4), -1000.0)
    kv_cur = torch.tensor([4.0, 0.0, 0.0, 8.0])
    score_cur = torch.zeros(4)
    ape = torch.zeros((4, 4))

    _next_kv, _next_score, emitted = compressor_update_state_reference(
        state_kv,
        state_score,
        kv_cur,
        score_cur,
        ape,
        token_idx=3,
        head_dim=2,
        ratio=4,
    )

    assert emitted is not None
    torch.testing.assert_close(emitted, torch.tensor([0.0, 8.0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_sliding_window_attention_matches_reference() -> None:
    from vllm.kernels.triton.deepseek_v4_flash.attention import (
        deepseek_v4_fused_sliding_window_attention,
    )

    num_heads = 64
    head_dim = 512
    window = 128
    torch.manual_seed(42)
    query = torch.randn((num_heads, head_dim), dtype=torch.float32, device="cuda")
    kv_rows = torch.randn((window, head_dim), dtype=torch.float32, device="cuda")
    attn_sinks = torch.randn((num_heads,), dtype=torch.float32, device="cuda")

    expected = shared_kv_swa_attention_reference(query, kv_rows, attn_sinks)
    actual = deepseek_v4_fused_sliding_window_attention(query, kv_rows, attn_sinks)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_sliding_window_attention_no_sinks() -> None:
    from vllm.kernels.triton.deepseek_v4_flash.attention import (
        deepseek_v4_fused_sliding_window_attention,
    )

    num_heads = 8
    head_dim = 32
    window = 7
    torch.manual_seed(7)
    query = torch.randn((num_heads, head_dim), dtype=torch.float32, device="cuda")
    kv_rows = torch.randn((window, head_dim), dtype=torch.float32, device="cuda")

    expected = shared_kv_swa_attention_reference(query, kv_rows, None)
    actual = deepseek_v4_fused_sliding_window_attention(query, kv_rows, None)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("layer_idx", [0, 1, 2, 3, 42])
@pytest.mark.parametrize("token_idx", [0, 1, 63, 127])
def test_precomputed_rope_matches_reference(
    layer_idx: int,
    token_idx: int,
) -> None:
    shape = DEEPSEEK_V4_FLASH_SHAPE
    rotary_dim = shape.rotary_dim
    context_length = 128
    device = torch.device("cpu")

    rope_cos, rope_sin = build_deepseek_layer_rope_tables(
        context_length=context_length,
        rotary_dim=rotary_dim,
        rope_freq_base=shape.rope_freq_base,
        compressed_rope_freq_base=shape.compressed_rope_freq_base,
        rope_scale_factor=shape.rope_scale_factor,
        rope_original_context=shape.rope_original_context,
        rope_yarn_beta_fast=shape.rope_yarn_beta_fast,
        rope_yarn_beta_slow=shape.rope_yarn_beta_slow,
        device=device,
    )

    torch.manual_seed(42)
    vectors_1d = torch.randn(rotary_dim, dtype=torch.float32, device=device)
    vectors_2d = torch.randn(4, rotary_dim, dtype=torch.float32, device=device)
    vectors_larger = torch.randn(rotary_dim + 16, dtype=torch.float32, device=device)

    cos = rope_cos[layer_idx, token_idx]
    sin = rope_sin[layer_idx, token_idx]

    for vectors in (vectors_1d, vectors_2d, vectors_larger):
        expected = apply_deepseek_layer_rope_to_tail_reference(
            vectors,
            token_idx=token_idx,
            layer_idx=layer_idx,
            rotary_dim=rotary_dim,
        )
        actual = apply_precomputed_rope_to_tail(vectors, cos, sin)
        torch.testing.assert_close(actual, expected)

        expected_inv = apply_deepseek_layer_rope_to_tail_reference(
            vectors,
            token_idx=token_idx,
            layer_idx=layer_idx,
            rotary_dim=rotary_dim,
            inverse=True,
        )
        actual_inv = apply_precomputed_rope_to_tail(
            vectors,
            cos,
            sin,
            inverse=True,
        )
        torch.testing.assert_close(actual_inv, expected_inv)
