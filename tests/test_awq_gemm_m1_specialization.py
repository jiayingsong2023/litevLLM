# SPDX-License-Identifier: Apache-2.0
"""Guardrails for M=1 decode on AWQ packed-int4 GEMM.

End-to-end Gemma4 decode currently keeps M=1 on deterministic heuristic tiles
and disables autotune, because autotune choices that look good in isolated
microbenchmarks have regressed whole-model decode latency.
"""
from __future__ import annotations

import os
from unittest import mock

import pytest
import torch

from vllm.kernels.triton.awq_fused_gemm import (
    _env_awq_o_proj_splitk_gemv,
    _env_awq_qo_proj_exact_gemv,
    _env_awq_o_proj_group32_gemv,
    _env_fused_gemm_autotune,
    packed_int4_symmetric_fused_gemm,
    packed_int4_symmetric_fused_qkv_m1_safe,
)
from vllm.model_executor.layers.quantization.tensor import (
    PackedInt4Weight,
    dequantize_symmetric_packed_int4_pytorch,
)


def test_env_autotune_never_true_for_m1_even_when_forced_on() -> None:
    with mock.patch.dict(os.environ, {"FASTINFERENCE_AWQ_FUSED_AUTOTUNE": "1"}):
        assert _env_fused_gemm_autotune(1, 8192, 8192) is False
        assert _env_fused_gemm_autotune(1, 43008, 5376) is False


def test_env_autotune_true_for_m2_when_forced_on() -> None:
    with mock.patch.dict(os.environ, {"FASTINFERENCE_AWQ_FUSED_AUTOTUNE": "1"}):
        assert _env_fused_gemm_autotune(2, 8192, 8192) is True


def test_o_proj_group32_gemv_default_enabled_and_overridable() -> None:
    with mock.patch.dict(os.environ, {}, clear=True):
        assert _env_awq_o_proj_group32_gemv() is True
    with mock.patch.dict(os.environ, {"FASTINFERENCE_AWQ_O_PROJ_GROUP32_GEMV": "0"}):
        assert _env_awq_o_proj_group32_gemv() is False


def test_qo_proj_exact_gemv_default_enabled_and_overridable() -> None:
    with mock.patch.dict(os.environ, {}, clear=True):
        assert _env_awq_qo_proj_exact_gemv() is False
    with mock.patch.dict(os.environ, {"FASTINFERENCE_AWQ_QO_PROJ_EXACT_GEMV": "1"}):
        assert _env_awq_qo_proj_exact_gemv() is True


def test_o_proj_splitk_gemv_default_disabled_and_overridable() -> None:
    with mock.patch.dict(os.environ, {}, clear=True):
        assert _env_awq_o_proj_splitk_gemv() is False
    with mock.patch.dict(os.environ, {"FASTINFERENCE_AWQ_O_PROJ_SPLITK_GEMV": "1"}):
        assert _env_awq_o_proj_splitk_gemv() is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm")
def test_packed_int4_m1_matches_dense_under_forced_autotune_env(monkeypatch) -> None:
    """Forced global autotune still keeps M=1 on deterministic MFMA heuristics."""
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
    """M=1 + split_k>1 exercises atomic reduction with MFMA tiles."""
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
    assert max_err < 20.0, f"max_err={max_err}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm")
def test_packed_int4_m1_grouped_decode_gemv_matches_dense(monkeypatch) -> None:
    """Env-gated grouped GEMV path matches dense dequant reference for M=1."""
    monkeypatch.setenv("FASTINFERENCE_AWQ_DECODE_GEMV", "1")
    monkeypatch.setenv("FASTINFERENCE_AWQ_FUSED_GEMM_SPLIT_K", "1")
    device = torch.device("cuda")
    torch.manual_seed(3)

    m, n, k = 1, 257, 288
    group_size = 32
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    qweight = torch.randint(0, 255, (n, k // 8), device=device, dtype=torch.uint8)
    scales = torch.randn((n, k // group_size), device=device, dtype=torch.float16).abs() + 0.01
    bias = torch.randn((n,), device=device, dtype=torch.float16)

    y_fused = packed_int4_symmetric_fused_gemm(x, qweight, scales, group_size, bias=bias)
    torch.cuda.synchronize()

    dense_weight = dequantize_symmetric_packed_int4_pytorch(
        qweight.to(torch.int32),
        scales,
        group_size=group_size,
    ).to(dtype=x.dtype)
    y_ref = torch.nn.functional.linear(x, dense_weight, bias.to(dtype=x.dtype))

    y_f = y_fused.float()
    y_r = y_ref.float()
    cos = float(
        ((y_f * y_r).sum() / (y_f.norm() * y_r.norm() + 1e-8)).item()
    )
    mae = float((y_f - y_r).abs().mean().item())
    max_err = float((y_f - y_r).abs().max().item())
    assert cos > 0.995, f"cos={cos}"
    assert mae < 0.18, f"mae={mae}"
    assert max_err < 2.5, f"max_err={max_err}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm")
def test_packed_int4_m1_grouped_decode_gemv_split_k_falls_back(monkeypatch) -> None:
    """Grouped GEMV is intentionally skipped for split-K until fp32 reduction exists."""
    monkeypatch.setenv("FASTINFERENCE_AWQ_DECODE_GEMV", "1")
    monkeypatch.setenv("FASTINFERENCE_AWQ_FUSED_GEMM_SPLIT_K", "4")
    device = torch.device("cuda")
    torch.manual_seed(4)

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
    assert max_err < 20.0, f"max_err={max_err}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm")
def test_packed_int4_m1_o_proj_group32_gemv_matches_dense(monkeypatch) -> None:
    """The 31B o_proj group32 GEMV specialization matches dense reference."""
    monkeypatch.setenv("FASTINFERENCE_AWQ_DECODE_GEMV", "1")
    monkeypatch.setenv("FASTINFERENCE_AWQ_FUSED_GEMM_SPLIT_K", "1")
    monkeypatch.setenv("FASTINFERENCE_AWQ_O_PROJ_GROUP32_GEMV", "1")
    device = torch.device("cuda")
    torch.manual_seed(5)

    m, n, k = 1, 5376, 16384
    group_size = 32
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    qweight = torch.randint(0, 255, (n, k // 8), device=device, dtype=torch.uint8)
    scales = torch.randn((n, k // group_size), device=device, dtype=torch.float16).abs() + 0.01

    y_fused = packed_int4_symmetric_fused_gemm(x, qweight, scales, group_size)
    torch.cuda.synchronize()
    monkeypatch.setenv("FASTINFERENCE_AWQ_O_PROJ_GROUP32_GEMV", "0")
    y_old = packed_int4_symmetric_fused_gemm(x, qweight, scales, group_size)
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
    rel_mae = mae / max(float(y_r.abs().mean().item()), 1e-6)
    old_mae = float((y_f - y_old.float()).abs().mean().item())
    old_max_err = float((y_f - y_old.float()).abs().max().item())
    assert cos > 0.995, f"cos={cos}"
    assert rel_mae < 0.004, f"mae={mae} rel_mae={rel_mae}"
    assert max_err < 20.0, f"max_err={max_err}"
    assert old_mae < 0.01, f"old_mae={old_mae}"
    assert old_max_err <= 1.0, f"old_max_err={old_max_err}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm")
def test_packed_int4_m1_o_proj_splitk_group32_gemv_matches_dense(monkeypatch) -> None:
    """The experimental o_proj split-K GEMV matches dense reference."""
    monkeypatch.setenv("FASTINFERENCE_AWQ_DECODE_GEMV", "1")
    monkeypatch.setenv("FASTINFERENCE_AWQ_FUSED_GEMM_SPLIT_K", "1")
    monkeypatch.setenv("FASTINFERENCE_AWQ_O_PROJ_SPLITK_GEMV", "1")
    monkeypatch.setenv("FASTINFERENCE_AWQ_O_PROJ_SPLITK", "4")
    device = torch.device("cuda")
    torch.manual_seed(12)

    m, n, k = 1, 5376, 16384
    group_size = 32
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    qweight = torch.randint(0, 255, (n, k // 8), device=device, dtype=torch.uint8)
    scales = torch.randn((n, k // group_size), device=device, dtype=torch.float16).abs() + 0.01

    y_splitk = packed_int4_symmetric_fused_gemm(x, qweight, scales, group_size)
    torch.cuda.synchronize()
    monkeypatch.setenv("FASTINFERENCE_AWQ_O_PROJ_SPLITK_GEMV", "0")
    y_old = packed_int4_symmetric_fused_gemm(x, qweight, scales, group_size)
    torch.cuda.synchronize()

    dense_weight = dequantize_symmetric_packed_int4_pytorch(
        qweight.to(torch.int32),
        scales,
        group_size=group_size,
    ).to(dtype=x.dtype)
    y_ref = torch.nn.functional.linear(x, dense_weight)

    y_s = y_splitk.float()
    y_r = y_ref.float()
    cos = float(
        ((y_s * y_r).sum() / (y_s.norm() * y_r.norm() + 1e-8)).item()
    )
    mae = float((y_s - y_r).abs().mean().item())
    max_err = float((y_s - y_r).abs().max().item())
    rel_mae = mae / max(float(y_r.abs().mean().item()), 1e-6)
    old_mae = float((y_s - y_old.float()).abs().mean().item())
    old_max_err = float((y_s - y_old.float()).abs().max().item())
    assert cos > 0.995, f"cos={cos}"
    assert rel_mae < 0.004, f"mae={mae} rel_mae={rel_mae}"
    assert max_err < 20.0, f"max_err={max_err}"
    assert old_mae < 0.02, f"old_mae={old_mae}"
    assert old_max_err <= 1.0, f"old_max_err={old_max_err}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm")
def test_packed_int4_m1_q_proj_exact_group32_gemv_matches_dense(monkeypatch) -> None:
    """The 31B q_proj exact-shape group32 GEMV matches dense reference."""
    monkeypatch.setenv("FASTINFERENCE_AWQ_DECODE_GEMV", "1")
    monkeypatch.setenv("FASTINFERENCE_AWQ_GROUP32_GEMV_ALL", "1")
    monkeypatch.setenv("FASTINFERENCE_AWQ_FUSED_GEMM_SPLIT_K", "1")
    monkeypatch.setenv("FASTINFERENCE_AWQ_QO_PROJ_EXACT_GEMV", "1")
    device = torch.device("cuda")
    torch.manual_seed(11)

    m, n, k = 1, 16384, 5376
    group_size = 32
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    qweight = torch.randint(0, 255, (n, k // 8), device=device, dtype=torch.uint8)
    scales = torch.randn((n, k // group_size), device=device, dtype=torch.float16).abs() + 0.01

    y_fused = packed_int4_symmetric_fused_gemm(x, qweight, scales, group_size)
    torch.cuda.synchronize()
    monkeypatch.setenv("FASTINFERENCE_AWQ_QO_PROJ_EXACT_GEMV", "0")
    y_old = packed_int4_symmetric_fused_gemm(x, qweight, scales, group_size)
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
    rel_mae = mae / max(float(y_r.abs().mean().item()), 1e-6)
    old_mae = float((y_f - y_old.float()).abs().mean().item())
    old_max_err = float((y_f - y_old.float()).abs().max().item())
    assert cos > 0.995, f"cos={cos}"
    assert rel_mae < 0.004, f"mae={mae} rel_mae={rel_mae}"
    assert max_err < 20.0, f"max_err={max_err}"
    assert old_mae < 0.01, f"old_mae={old_mae}"
    assert old_max_err <= 1.0, f"old_max_err={old_max_err}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm")
def test_packed_int4_m1_qkv_group32_gemv_matches_three_gemvs(monkeypatch) -> None:
    """Fused local q/k/v decode matches the existing per-projection GEMV path."""
    monkeypatch.setenv("FASTINFERENCE_AWQ_DECODE_GEMV", "1")
    monkeypatch.setenv("FASTINFERENCE_AWQ_FUSED_GEMM_SPLIT_K", "1")
    device = torch.device("cuda")
    torch.manual_seed(6)

    m, k = 1, 5376
    qn, kn, vn = 8192, 4096, 4096
    group_size = 32
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    qweight = torch.randint(0, 255, (qn, k // 8), device=device, dtype=torch.int32)
    kweight = torch.randint(0, 255, (kn, k // 8), device=device, dtype=torch.int32)
    vweight = torch.randint(0, 255, (vn, k // 8), device=device, dtype=torch.int32)
    qscales = torch.randn((qn, k // group_size), device=device, dtype=torch.float16).abs() + 0.01
    kscales = torch.randn((kn, k // group_size), device=device, dtype=torch.float16).abs() + 0.01
    vscales = torch.randn((vn, k // group_size), device=device, dtype=torch.float16).abs() + 0.01

    y_fused, used, reason = packed_int4_symmetric_fused_qkv_m1_safe(
        x, qweight, kweight, vweight, qscales, kscales, vscales, group_size
    )
    assert used, reason
    torch.cuda.synchronize()

    monkeypatch.setenv("FASTINFERENCE_AWQ_QO_PROJ_EXACT_GEMV", "0")
    yq = packed_int4_symmetric_fused_gemm(x, qweight, qscales, group_size)
    yk = packed_int4_symmetric_fused_gemm(x, kweight, kscales, group_size)
    yv = packed_int4_symmetric_fused_gemm(x, vweight, vscales, group_size)
    torch.cuda.synchronize()
    y_ref = torch.cat([yq, yk, yv], dim=1)

    diff = (y_fused.float() - y_ref.float()).abs()
    assert float(diff.mean().item()) < 0.001
    assert float(diff.max().item()) <= 4.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm")
def test_packed_int4_m1_qk_group32_gemv_matches_two_gemvs(monkeypatch) -> None:
    """Fused global q/k decode handles Gemma4 attention_k_eq_v layers."""
    monkeypatch.setenv("FASTINFERENCE_AWQ_DECODE_GEMV", "1")
    monkeypatch.setenv("FASTINFERENCE_AWQ_FUSED_GEMM_SPLIT_K", "1")
    device = torch.device("cuda")
    torch.manual_seed(7)

    m, k = 1, 5376
    qn, kn = 16384, 2048
    group_size = 32
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    qweight = torch.randint(0, 255, (qn, k // 8), device=device, dtype=torch.int32)
    kweight = torch.randint(0, 255, (kn, k // 8), device=device, dtype=torch.int32)
    qscales = torch.randn((qn, k // group_size), device=device, dtype=torch.float16).abs() + 0.01
    kscales = torch.randn((kn, k // group_size), device=device, dtype=torch.float16).abs() + 0.01

    y_fused, used, reason = packed_int4_symmetric_fused_qkv_m1_safe(
        x, qweight, kweight, None, qscales, kscales, None, group_size
    )
    assert used, reason
    torch.cuda.synchronize()

    monkeypatch.setenv("FASTINFERENCE_AWQ_QO_PROJ_EXACT_GEMV", "0")
    yq = packed_int4_symmetric_fused_gemm(x, qweight, qscales, group_size)
    yk = packed_int4_symmetric_fused_gemm(x, kweight, kscales, group_size)
    torch.cuda.synchronize()
    y_ref = torch.cat([yq, yk], dim=1)

    diff = (y_fused.float() - y_ref.float()).abs()
    assert float(diff.mean().item()) < 0.001
    assert float(diff.max().item()) <= 0.01


def test_gemma4_down_proj_decode_bypasses_dense_fallback_when_gemv_enabled(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_DENSE_DOWN_PROJ", "1")
    monkeypatch.setenv("FASTINFERENCE_AWQ_DECODE_GEMV", "1")

    qweight = torch.zeros((5376, 2688), dtype=torch.uint8)
    scales = torch.ones((5376, 672), dtype=torch.float16)
    weight = PackedInt4Weight(
        qweight,
        scales,
        group_size=32,
        original_shape=(5376, 21504),
        prefix="layers.0.mlp.down_proj",
        profile_hint="gemma4_31b_q4",
    )
    x = torch.zeros((1, 21504), dtype=torch.bfloat16)
    expected = torch.full((1, 5376), 0.25, dtype=torch.bfloat16)

    def _fake_safe(a, qweight, scales, group_size, out=None, bias=None):
        return expected, True, "ok"

    monkeypatch.setattr(
        "vllm.kernels.triton.awq_fused_gemm.packed_int4_symmetric_fused_gemm_safe",
        _fake_safe,
    )

    def _raise_dense(*args, **kwargs):
        raise AssertionError("dense fallback should be bypassed for gemma4 down_proj decode")

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.tensor.dequantize_symmetric_packed_int4_pytorch",
        _raise_dense,
    )

    y = weight.matmul(x)
    assert y.shape == (1, 5376)
    assert torch.equal(y, expected)
