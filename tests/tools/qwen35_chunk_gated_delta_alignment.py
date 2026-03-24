#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Compare FastInference ``_torch_chunk_gated_delta_rule`` against
``fla.ops.gated_delta_rule.naive.naive_chunk_gated_delta_rule`` (CPU reference).

The pure PyTorch path matches the FLA naive reference (bit-for-bit on CPU).
Optional: ``FASTINFERENCE_QWEN35_USE_FLA_CHUNK=1`` uses the fused CUDA kernel from
``flash-linear-attention`` (small numeric differences vs naive are possible).

Usage:
  uv run python tests/tools/qwen35_chunk_gated_delta_alignment.py
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _pair_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    x = a.float().reshape(-1)
    y = b.float().reshape(-1)
    n = min(x.numel(), y.numel())
    diff = (x[:n] - y[:n]).abs()
    cos = F.cosine_similarity(x[:n].unsqueeze(0), y[:n].unsqueeze(0), dim=1).item()
    return {"cos_sim": cos, "max_err": diff.max().item(), "mae": diff.mean().item()}


def main() -> int:
    from vllm.model_executor.models.qwen3_5 import _l2norm, _torch_chunk_gated_delta_rule

    try:
        from fla.ops.gated_delta_rule.naive import naive_chunk_gated_delta_rule
    except ImportError:
        print("Install flash-linear-attention for this check: uv pip install flash-linear-attention")
        return 1

    torch.manual_seed(0)
    B, T, H, K, V = 2, 100, 4, 32, 48
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    beta = torch.sigmoid(torch.randn(B, T, H))
    g = -torch.randn(B, T, H).abs()

    out_lite, _ = _torch_chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta, chunk_size=64, use_qk_l2norm_in_kernel=False
    )
    out_naive, _ = naive_chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=64)
    m = _pair_metrics(out_lite, out_naive)
    print("=== Lite vs FLA naive (no in-kernel L2) ===")
    print(f"  CosSim={m['cos_sim']:.8f}  max_err={m['max_err']:.6e}  mae={m['mae']:.6e}")
    ok = m["max_err"] == 0.0

    out_l2, _ = _torch_chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta, chunk_size=64, use_qk_l2norm_in_kernel=True
    )
    qn = _l2norm(q, dim=-1)
    kn = _l2norm(k, dim=-1)
    out_ext, _ = naive_chunk_gated_delta_rule(qn, kn, v, g, beta, chunk_size=64)
    m2 = _pair_metrics(out_l2, out_ext)
    print("=== in-kernel L2 vs external L2 + naive ===")
    print(f"  CosSim={m2['cos_sim']:.8f}  max_err={m2['max_err']:.6e}  mae={m2['mae']:.6e}")
    ok = ok and m2["max_err"] == 0.0

    if torch.cuda.is_available():
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        except ImportError:
            ...
        else:
            dev = torch.device("cuda")
            qd = q.to(dev)
            kd = k.to(dev)
            vd = v.to(dev)
            gd = g.to(dev)
            bd = beta.to(dev)
            out_lite_d, _ = _torch_chunk_gated_delta_rule(
                qd, kd, vd, g=gd, beta=bd, chunk_size=64, use_qk_l2norm_in_kernel=False
            )
            out_fla_d, _ = chunk_gated_delta_rule(
                qd,
                kd,
                vd,
                gd,
                bd,
                scale=K**-0.5,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
            )
            m3 = _pair_metrics(out_lite_d.cpu(), out_fla_d.cpu())
            print("=== CUDA: Lite (torch ref) vs FLA fused chunk_gated_delta_rule / no L2 ===")
            print(f"  CosSim={m3['cos_sim']:.8f}  max_err={m3['max_err']:.6e}  mae={m3['mae']:.6e}")
            print(
                "  (Fused Triton kernels are not bitwise-identical to the naive reference; "
                "expect small element-wise drift vs the pure PyTorch path.)"
            )
    else:
        print("=== CUDA: skipped (no GPU) ===")

    print(f"  status: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
