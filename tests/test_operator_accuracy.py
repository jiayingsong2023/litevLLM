# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn.functional as F
import pytest

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.layers.quantization.tensor import dequantize_awq_pytorch
from vllm.kernels.triton.paged_attention import paged_attention_v1
from vllm.kernels.triton.reshape_and_cache import reshape_and_cache


def _dequantize_awq_reference(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Reference implementation with explicit loops for correctness checks."""
    n_rows, n_cols_packed = qweight.shape
    n_cols = n_cols_packed * 8
    n_groups = n_cols // group_size
    out = torch.empty((n_rows, n_cols), dtype=torch.float32)

    for r in range(n_rows):
        for c in range(n_cols):
            packed_col = c // 8
            nibble_idx = c % 8
            shift = nibble_idx * 4
            qv = int((int(qweight[r, packed_col]) >> shift) & 0x0F)

            g = c // group_size
            packed_g = g // 8
            nibble_g = g % 8
            shift_g = nibble_g * 4
            z = int((int(qzeros[r, packed_g]) >> shift_g) & 0x0F)
            s = float(scales[r, g])
            out[r, c] = (qv - z) * s

    return out.to(torch.float16)


def test_awq_dequant_matches_reference_group32():
    torch.manual_seed(7)
    n_rows = 6
    n_cols = 256
    group_size = 32
    n_groups = n_cols // group_size

    qweight = torch.randint(
        low=0,
        high=2**31 - 1,
        size=(n_rows, n_cols // 8),
        dtype=torch.int32,
    )
    scales = torch.rand((n_rows, n_groups), dtype=torch.float16) * 0.5 + 0.01
    qzeros = torch.randint(
        low=0,
        high=2**31 - 1,
        size=(n_rows, n_groups // 8),
        dtype=torch.int32,
    )

    got = dequantize_awq_pytorch(qweight, scales, qzeros, group_size=group_size)
    expected = _dequantize_awq_reference(qweight, scales, qzeros, group_size)
    assert torch.equal(got, expected)


def test_lite_linear_matches_torch_linear():
    torch.manual_seed(11)
    in_features = 16
    out_features = 10
    batch = 5

    layer = LiteLinear(in_features, out_features, bias=True)
    weight = torch.randn((out_features, in_features), dtype=torch.float32)
    bias = torch.randn((out_features,), dtype=torch.float32)
    layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
    layer.bias = torch.nn.Parameter(bias.clone(), requires_grad=False)

    x = torch.randn((batch, in_features), dtype=torch.float32)
    got = layer(x)
    expected = F.linear(x, weight, bias)
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-6)


def test_rmsnorm_matches_reference():
    torch.manual_seed(23)
    hidden = 32
    eps = 1e-6
    x = torch.randn((4, 3, hidden), dtype=torch.float32)

    norm = RMSNorm(hidden_size=hidden, eps=eps)
    norm.weight = torch.nn.Parameter(torch.randn(hidden), requires_grad=False)

    got = norm(x)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    expected = norm.weight * (x * torch.rsqrt(var + eps))
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-6)


def _reference_causal_attention_gqa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> torch.Tensor:
    """
    q: [T, Hq, D], k/v: [T, Hkv, D]
    Returns causal attention output: [T, Hq, D]
    """
    t_len, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_heads // num_kv_heads
    out = torch.empty_like(q, dtype=torch.float32)
    for t in range(t_len):
        for h in range(num_heads):
            kvh = h // group_size
            qh = q[t, h].float()  # [D]
            ks = k[: t + 1, kvh].float()  # [t+1, D]
            vs = v[: t + 1, kvh].float()  # [t+1, D]
            scores = (ks @ qh) * scale
            probs = torch.softmax(scores, dim=0)
            out[t, h] = probs.unsqueeze(0) @ vs
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA/ROCm device")
def test_paged_attention_prefill_matches_reference_gqa():
    torch.manual_seed(29)
    device = "cuda"
    dtype = torch.float16
    seq_len = 5
    num_heads = 8
    num_kv_heads = 2
    head_dim = 16
    block_size = 16
    num_blocks_per_seq = 4
    num_total_blocks = num_blocks_per_seq

    # Token-wise q/k/v for one request.
    q = torch.randn((seq_len, num_heads, head_dim), device=device, dtype=dtype)
    k = torch.randn((seq_len, num_kv_heads, head_dim), device=device, dtype=dtype)
    v = torch.randn((seq_len, num_kv_heads, head_dim), device=device, dtype=dtype)

    k_cache = torch.zeros(
        (num_total_blocks, block_size, num_kv_heads, head_dim),
        device=device,
        dtype=dtype,
    )
    v_cache = torch.zeros_like(k_cache)
    slot_mapping = torch.arange(seq_len, device=device, dtype=torch.long)
    reshape_and_cache(k.contiguous(), v.contiguous(), k_cache, v_cache, slot_mapping,
                      "auto")

    # Prefill path in llama/qwen full attention expands metadata per token.
    block_table = torch.arange(num_blocks_per_seq, device=device,
                               dtype=torch.int32).unsqueeze(0)
    block_tables_ext = block_table.expand(seq_len, -1).contiguous()
    seq_lens_ext = torch.arange(1, seq_len + 1, device=device, dtype=torch.int32)

    out = torch.empty((seq_len, num_heads, head_dim), device=device, dtype=dtype)
    scale = head_dim**-0.5
    paged_attention_v1(
        out,
        q.contiguous(),
        k_cache,
        v_cache,
        num_heads,
        scale,
        block_tables_ext,
        seq_lens_ext,
        block_size,
        4096,
        None,
        "auto",
        None,
        None,
        num_kv_heads=num_kv_heads,
    )

    ref = _reference_causal_attention_gqa(q, k, v, scale)
    assert torch.allclose(out.float(), ref, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA/ROCm device")
def test_paged_attention_decode_matches_reference_gqa():
    torch.manual_seed(31)
    device = "cuda"
    dtype = torch.float16
    seq_len = 6
    num_heads = 8
    num_kv_heads = 2
    head_dim = 16
    block_size = 16
    num_blocks_per_seq = 4
    num_total_blocks = num_blocks_per_seq

    q_all = torch.randn((seq_len, num_heads, head_dim), device=device, dtype=dtype)
    k_all = torch.randn((seq_len, num_kv_heads, head_dim), device=device, dtype=dtype)
    v_all = torch.randn((seq_len, num_kv_heads, head_dim), device=device, dtype=dtype)

    # Build cache with all tokens written in physical order.
    k_cache = torch.zeros(
        (num_total_blocks, block_size, num_kv_heads, head_dim),
        device=device,
        dtype=dtype,
    )
    v_cache = torch.zeros_like(k_cache)
    slot_mapping = torch.arange(seq_len, device=device, dtype=torch.long)
    reshape_and_cache(k_all.contiguous(), v_all.contiguous(), k_cache, v_cache,
                      slot_mapping, "auto")

    # Decode path: query only the last token, seq_lens=[full_len].
    q_last = q_all[-1:].contiguous()
    out = torch.empty((1, num_heads, head_dim), device=device, dtype=dtype)
    block_table = torch.arange(num_blocks_per_seq, device=device,
                               dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)
    scale = head_dim**-0.5
    paged_attention_v1(
        out,
        q_last,
        k_cache,
        v_cache,
        num_heads,
        scale,
        block_table,
        seq_lens,
        block_size,
        4096,
        None,
        "auto",
        None,
        None,
        num_kv_heads=num_kv_heads,
    )

    ref_all = _reference_causal_attention_gqa(q_all, k_all, v_all, scale)
    ref_last = ref_all[-1:]
    assert torch.allclose(out.float(), ref_last, atol=3e-2, rtol=3e-2)
