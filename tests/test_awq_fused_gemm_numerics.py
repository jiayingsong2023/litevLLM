# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from vllm.kernels.triton.awq_fused_gemm import packed_int4_symmetric_fused_gemm
from vllm.model_executor.layers.quantization.tensor import (
    dequantize_symmetric_packed_int4_pytorch,
)


def _cosine_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    num = (a * b).sum(dim=-1)
    den = a.norm(dim=-1) * b.norm(dim=-1) + 1e-8
    return float((num / den).mean().item())


def test_symmetric_packed_int4_dequant_preserves_expert_batch() -> None:
    qweight = torch.tensor(
        [
            [[0x76543210], [0x6EDCBA98]],
            [[0x01234567], [0x09ABCDEF]],
        ],
        dtype=torch.int32,
    )
    scales = torch.tensor(
        [
            [[0.5], [0.25]],
            [[0.125], [0.75]],
        ],
        dtype=torch.float16,
    )

    batched = dequantize_symmetric_packed_int4_pytorch(
        qweight,
        scales,
        group_size=8,
    )
    stacked = torch.stack(
        [
            dequantize_symmetric_packed_int4_pytorch(
                qweight[expert],
                scales[expert],
                group_size=8,
            )
            for expert in range(int(qweight.shape[0]))
        ],
        dim=0,
    )

    assert tuple(batched.shape) == (2, 2, 8)
    torch.testing.assert_close(batched, stacked)


def test_packed_int4_fused_matches_dense_when_block_spans_multiple_groups(
    monkeypatch,
) -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    torch.manual_seed(0)

    # Force the heuristic kernel path (not autotune/split-k) with BLOCK_K=64.
    monkeypatch.setenv("FASTINFERENCE_AWQ_FUSED_AUTOTUNE", "0")
    monkeypatch.setenv("FASTINFERENCE_AWQ_FUSED_GEMM_SPLIT_K", "1")
    monkeypatch.setenv("FASTINFERENCE_AWQ_FUSED_GEMM_BLOCK_M", "16")
    monkeypatch.setenv("FASTINFERENCE_AWQ_FUSED_GEMM_BLOCK_N", "64")
    monkeypatch.setenv("FASTINFERENCE_AWQ_FUSED_GEMM_BLOCK_K", "64")
    monkeypatch.setenv("FASTINFERENCE_AWQ_FUSED_GEMM_NUM_WARPS", "4")
    monkeypatch.setenv("FASTINFERENCE_AWQ_FUSED_GEMM_NUM_STAGES", "2")

    # group_size(32) < BLOCK_K(64): this is the historical mismatch pattern.
    m, n, k = 4, 128, 128
    group_size = 32
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    qweight = torch.randint(0, 255, (n, k // 8), device=device, dtype=torch.uint8)
    scales = (
        torch.randn((n, k // group_size), device=device, dtype=torch.float16).abs()
        + 0.01
    )

    y_fused = packed_int4_symmetric_fused_gemm(x, qweight, scales, group_size)

    dense_weight = dequantize_symmetric_packed_int4_pytorch(
        qweight.to(torch.int32),
        scales,
        group_size=group_size,
    ).to(dtype=x.dtype)
    y_dense = torch.nn.functional.linear(x, dense_weight)

    y_f32 = y_fused.float()
    y_ref = y_dense.float()
    cos = _cosine_mean(y_f32, y_ref)
    mae = float((y_f32 - y_ref).abs().mean().item())
    max_err = float((y_f32 - y_ref).abs().max().item())

    # Old buggy kernel path showed significantly worse alignment (cos around 0.95
    # on Gemma4-like shapes). Keep thresholds loose enough for bf16 rounding, but
    # tight enough to catch group-scale indexing regressions.
    assert cos > 0.995
    assert mae < 0.2
    assert max_err < 2.0
