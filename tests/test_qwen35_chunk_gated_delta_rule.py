# SPDX-License-Identifier: Apache-2.0
"""Regression: Lite chunk gated delta rule vs flash-linear-attention reference."""

import pytest
import torch

from vllm.model_executor.models.qwen3_5 import _torch_chunk_gated_delta_rule


def _require_fla_naive():
    return pytest.importorskip(
        "fla.ops.gated_delta_rule.naive",
        reason="pip install flash-linear-attention to run this test",
    )


@pytest.mark.parametrize("T", [17, 64, 100])
def test_lite_matches_fla_naive(T: int):
    naive_mod = _require_fla_naive()
    naive_chunk_gated_delta_rule = naive_mod.naive_chunk_gated_delta_rule

    torch.manual_seed(42)
    B, H, K, V = 2, 4, 32, 48
    q = torch.randn(B, T, H, K, dtype=torch.float32)
    k = torch.randn(B, T, H, K, dtype=torch.float32)
    v = torch.randn(B, T, H, V, dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.float32))
    g = -torch.randn(B, T, H, dtype=torch.float32).abs()

    out_lite, _ = _torch_chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        chunk_size=64,
        use_qk_l2norm_in_kernel=False,
    )
    out_ref, _ = naive_chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=64)
    assert torch.equal(out_lite, out_ref)


def test_lite_matches_fla_naive_with_initial_state():
    naive_mod = _require_fla_naive()
    naive_chunk_gated_delta_rule = naive_mod.naive_chunk_gated_delta_rule

    torch.manual_seed(7)
    B, T, H, K, V = 2, 77, 4, 32, 48
    q = torch.randn(B, T, H, K, dtype=torch.float32)
    k = torch.randn(B, T, H, K, dtype=torch.float32)
    v = torch.randn(B, T, H, V, dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.float32))
    g = -torch.randn(B, T, H, dtype=torch.float32).abs()
    h0 = torch.randn(B, H, K, V, dtype=torch.float32)

    out_lite, st_lite = _torch_chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        chunk_size=64,
        initial_state=h0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )
    out_ref, st_ref = naive_chunk_gated_delta_rule(
        q, k, v, g, beta, chunk_size=64, initial_state=h0, output_final_state=True
    )
    assert torch.equal(out_lite, out_ref)
    assert torch.equal(st_lite, st_ref)


def test_use_qk_l2norm_in_kernel_matches_external_l2():
    """Same as applying L2 norm to q/k before the rule without in-kernel flag."""
    naive_mod = _require_fla_naive()
    naive_chunk_gated_delta_rule = naive_mod.naive_chunk_gated_delta_rule
    from vllm.model_executor.models.qwen3_5 import _l2norm

    torch.manual_seed(0)
    B, T, H, K, V = 1, 32, 2, 16, 24
    q = torch.randn(B, T, H, K, dtype=torch.float32)
    k = torch.randn(B, T, H, K, dtype=torch.float32)
    v = torch.randn(B, T, H, V, dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.float32))
    g = -torch.randn(B, T, H, dtype=torch.float32).abs()

    out_in, _ = _torch_chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        chunk_size=64,
        use_qk_l2norm_in_kernel=True,
    )
    qn = _l2norm(q, dim=-1)
    kn = _l2norm(k, dim=-1)
    out_ext, _ = naive_chunk_gated_delta_rule(qn, kn, v, g, beta, chunk_size=64)
    assert torch.equal(out_in, out_ext)
