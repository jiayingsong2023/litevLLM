# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm.kernels.triton.gemma4_moe_int4 import gemma4_moe_int4_decode_batched_chunked
from vllm.model_executor.layers.quantization.tensor import (
    dequantize_symmetric_packed_int4_pytorch,
)


def _reference_moe_single(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    qweight_gu: torch.Tensor,
    scales_gu: torch.Tensor,
    qweight_d: torch.Tensor,
    scales_d: torch.Tensor,
    intermediate_dim: int,
) -> torch.Tensor:
    m, hidden = x.shape
    top_k = topk_ids.shape[1]
    out = torch.zeros(m, hidden, dtype=torch.float32, device=x.device)
    for tok in range(m):
        for k in range(top_k):
            eid = int(topk_ids[tok, k])
            gsz_gu = hidden // scales_gu.shape[2]
            gsz_d = intermediate_dim // scales_d.shape[2]
            w1e = dequantize_symmetric_packed_int4_pytorch(
                qweight_gu[eid : eid + 1].to(torch.int32),
                scales_gu[eid : eid + 1],
                group_size=gsz_gu,
            )[0, : 2 * intermediate_dim, :hidden].to(torch.float32)
            w2e = dequantize_symmetric_packed_int4_pytorch(
                qweight_d[eid : eid + 1].to(torch.int32),
                scales_d[eid : eid + 1],
                group_size=gsz_d,
            )[0, :hidden, :intermediate_dim].to(torch.float32)

            gu = torch.matmul(x[tok : tok + 1].to(torch.float32), w1e.t())
            g, u = gu.chunk(2, dim=-1)
            h = torch.nn.functional.silu(g) * u
            y = torch.matmul(h, w2e.t()) * topk_weights[tok, k].to(torch.float32)
            out[tok] += y.squeeze(0)
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_moe_int4_numerical_small_shape() -> None:
    # Use a small shape for fast reference comparison. The microbench script
    # uses the real 26B shape (hidden=2816, intermediate=704); this test only
    # verifies numerical correctness of the kernel path.
    device = torch.device("cuda")
    hidden_dim = 256
    intermediate_dim = 64
    num_experts = 16
    top_k = 4
    m = 2

    torch.manual_seed(42)
    x = torch.randn(m, hidden_dim, dtype=torch.bfloat16, device=device)
    topk_ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], device=device)
    topk_weights = torch.full(
        (m, top_k), 1.0 / top_k, dtype=torch.bfloat16, device=device
    )

    qweight_gu = torch.randint(
        0,
        16,
        (num_experts, 2 * intermediate_dim, hidden_dim // 8),
        dtype=torch.int32,
        device=device,
    )
    scales_gu = torch.randn(
        num_experts,
        2 * intermediate_dim,
        hidden_dim // 32,
        dtype=torch.bfloat16,
        device=device,
    )
    qweight_d = torch.randint(
        0,
        16,
        (num_experts, hidden_dim, intermediate_dim // 8),
        dtype=torch.int32,
        device=device,
    )
    scales_d = torch.randn(
        num_experts,
        hidden_dim,
        intermediate_dim // 32,
        dtype=torch.bfloat16,
        device=device,
    )

    out, used, reason = gemma4_moe_int4_decode_batched_chunked(
        x,
        topk_weights,
        topk_ids,
        qweight_gu,
        scales_gu,
        qweight_d,
        scales_d,
        intermediate_dim=intermediate_dim,
        activation="silu",
    )
    assert used, f"Fast path not used: {reason}"
    assert out.shape == (m, hidden_dim)
    assert out.dtype == x.dtype

    ref = _reference_moe_single(
        x,
        topk_ids,
        topk_weights,
        qweight_gu,
        scales_gu,
        qweight_d,
        scales_d,
        intermediate_dim,
    )
    torch.testing.assert_close(out.to(torch.float32), ref, rtol=3e-2, atol=3e-2)
