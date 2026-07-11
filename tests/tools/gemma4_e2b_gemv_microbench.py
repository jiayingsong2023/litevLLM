#!/usr/bin/env python3
"""Microbench asymmetric GEMV vs dense reference for E2B shapes."""

import os
import sys
import time

# Repo root so ``from tests.tools...`` works when running this script directly.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402

from tests.tools.gemma4_e2b_quant_helpers import (  # noqa: E402
    make_asymmetric_packed_int4,
)
from vllm.kernels.triton.awq_fused_gemm import (  # noqa: E402
    packed_int4_asymmetric_group32_gemv_m1,
)
from vllm.model_executor.layers.quantization.tensor import (  # noqa: E402
    dequantize_asymmetric_packed_int4_pytorch,
)

SHAPES = [
    (2048, 1536),
    (6144, 1536),
    (1536, 6144),
    (1536, 2048),
    # Actual E2B PLE projection shape: N=num_layers*ple_dim, K=hidden_size.
    (35 * 256, 1536),
]


def bench():
    for n, k in SHAPES:
        qweight, scales, qzeros = make_asymmetric_packed_int4(n, k, 32)
        x = torch.randn(1, k, dtype=torch.float16, device="cuda")
        dense = dequantize_asymmetric_packed_int4_pytorch(qweight, scales, qzeros, 32)

        for _ in range(10):
            packed_int4_asymmetric_group32_gemv_m1(x, qweight, scales, qzeros)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(1000):
            packed_int4_asymmetric_group32_gemv_m1(x, qweight, scales, qzeros)
        torch.cuda.synchronize()
        fused_ms = (time.perf_counter() - t0) / 1000 * 1000

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(1000):
            torch.nn.functional.linear(x, dense)
        torch.cuda.synchronize()
        dense_ms = (time.perf_counter() - t0) / 1000 * 1000

        print(
            f"N={n:5d} K={k:5d} fused={fused_ms:.4f}ms "
            f"dense={dense_ms:.4f}ms speedup={dense_ms / fused_ms:.2f}x"
        )


if __name__ == "__main__":
    bench()
