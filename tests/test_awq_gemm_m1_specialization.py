# SPDX-License-Identifier: Apache-2.0
"""Stage 6 / Step 6-A guardrails for M=1 decode on AWQ packed-int4 GEMM.

The Triton autotune table only lists ``BLOCK_M >= 16``. When M==1 at
runtime those tiles still mask to a single logical row, but
``BLOCK_M == 1`` is false at compile time so the in-kernel GEMV branch
(``tl.sum`` over K) never fires and ``tl.dot`` wastes MFMA lanes.

``_env_fused_gemm_autotune`` must therefore return False for M==1 even
when ``FASTINFERENCE_AWQ_FUSED_AUTOTUNE=1``. This file locks that
contract and checks ``packed_int4_symmetric_fused_gemm`` against a dense
PyTorch reference for M=1 under forced-autotune env.
"""
from __future__ import annotations

import os
from unittest import mock

import pytest
import torch

from vllm.kernels.triton.awq_fused_gemm import (
    _env_fused_gemm_autotune,
    packed_int4_symmetric_fused_gemm,
)
from vllm.model_executor.layers.quantization.tensor import (
    dequantize_symmetric_packed_int4_pytorch,
)


def test_env_autotune_never_true_for_m1_even_when_forced_on() -> None:
    with mock.patch.dict(os.environ, {"FASTINFERENCE_AWQ_FUSED_AUTOTUNE": "1"}):
        assert _env_fused_gemm_autotune(1, 8192, 8192) is False
        assert _env_fused_gemm_autotune(1, 43008, 5376) is False


def test_env_autotune_true_for_m2_when_forced_on() -> None:
    with mock.patch.dict(os.environ, {"FASTINFERENCE_AWQ_FUSED_AUTOTUNE": "1"}):
        assert _env_fused_gemm_autotune(2, 8192, 8192) is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm")
def test_packed_int4_m1_matches_dense_under_forced_autotune_env(monkeypatch) -> None:
    """Regression: forced global autotune must not route M=1 through autotuned
    configs (BLOCK_M>=16), which would skip the GEMV fast path.
    """
    monkeypatch.setenv("FASTINFERENCE_AWQ_FUSED_AUTOTUNE", "1")
    monkeypatch.setenv("FASTINFERENCE_AWQ_FUSED_GEMM_SPLIT_K", "1")
    device = torch.device("cuda")
    torch.manual_seed(1)

    m, n, k = 1, 192, 256
    group_size = 32
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    qweight = torch.randint(0, 255, (n, k // 8), device=device, dtype=torch.uint8)
    scales = torch.randn((n, k // group_size), device=device, dtype=torch.float16).abs() + 0.01

    y_fused = packed_int4_symmetric_fused_gemm(x, qweight, scales, group_size)
    torch.cuda.synchronize()

    dense_weight = dequantize_symmetric_packed_int4_pytorch(
        qweight.to(torch.int32),
        scales,
        group_size=group_size,
    ).to(dtype=x.dtype)
    y_ref = torch.nn.functional.linear(x, dense_weight)

    y_f = y_fused.float()
    y_r = y_ref.float()
    cos = float(
        ((y_f * y_r).sum() / (y_f.norm() * y_r.norm() + 1e-8)).item()
    )
    mae = float((y_f - y_r).abs().mean().item())
    assert cos > 0.995, f"cos={cos}"
    assert mae < 0.15, f"mae={mae}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm")
def test_packed_int4_m1_split_k_matches_dense(monkeypatch) -> None:
    """M=1 + split_k>1 exercises atomic reduction with BLOCK_M==1 GEMV inner."""
    monkeypatch.setenv("FASTINFERENCE_AWQ_FUSED_AUTOTUNE", "0")
    monkeypatch.setenv("FASTINFERENCE_AWQ_FUSED_GEMM_SPLIT_K", "4")
    device = torch.device("cuda")
    torch.manual_seed(2)

    m, n, k = 1, 96, 256
    group_size = 32
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    qweight = torch.randint(0, 255, (n, k // 8), device=device, dtype=torch.uint8)
    scales = torch.randn((n, k // group_size), device=device, dtype=torch.float16).abs() + 0.01

    y_fused = packed_int4_symmetric_fused_gemm(x, qweight, scales, group_size)
    torch.cuda.synchronize()

    dense_weight = dequantize_symmetric_packed_int4_pytorch(
        qweight.to(torch.int32),
        scales,
        group_size=group_size,
    ).to(dtype=x.dtype)
    y_ref = torch.nn.functional.linear(x, dense_weight)

    y_f = y_fused.float()
    y_r = y_ref.float()
    cos = float(
        ((y_f * y_r).sum() / (y_f.norm() * y_r.norm() + 1e-8)).item()
    )
    mae = float((y_f - y_r).abs().mean().item())
    max_err = float((y_f - y_r).abs().max().item())
    assert cos > 0.99, f"cos={cos}"
    assert mae < 0.28, f"mae={mae}"
    assert max_err < 3.0, f"max_err={max_err}"
