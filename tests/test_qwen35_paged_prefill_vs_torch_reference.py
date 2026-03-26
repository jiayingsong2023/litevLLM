# SPDX-License-Identifier: Apache-2.0
"""Regression: Lite paged attention prefill vs reference causal softmax (CUDA).

Mirrors Qwen3_5FullAttentionLayer prefill branch: per-row seq_lens[i] = start_pos + i + 1
with start_pos=0, block_tables shared across rows.
"""

import os

import pytest
import torch
import torch.nn.functional as F

# FP16 KV cache avoids FP8 in reshape_and_cache for this test harness.
os.environ.setdefault("FASTINFERENCE_KV_FP8", "1")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_paged_prefill_matches_causal_softmax_reference():
    from vllm.kernels.triton.paged_attention import paged_attention_v1
    from vllm.kernels.triton.reshape_and_cache import reshape_and_cache

    device = torch.device("cuda:0")
    torch.manual_seed(0)
    block_size = 16
    nh, nkv, hd = 4, 2, 8
    group = nh // nkv
    assert nh % nkv == 0
    S = 4
    num_physical_blocks = 1

    k_cache = torch.zeros(
        (num_physical_blocks, block_size, nkv, hd), device=device, dtype=torch.float16
    )
    v_cache = torch.zeros_like(k_cache)
    k = torch.randn(S, nkv, hd, device=device, dtype=torch.float16)
    v = torch.randn(S, nkv, hd, device=device, dtype=torch.float16)
    q = torch.randn(S, nh, hd, device=device, dtype=torch.float16)

    slot_mapping = torch.arange(S, device=device, dtype=torch.long)
    reshape_and_cache(k, v, k_cache, v_cache, slot_mapping, "auto")

    scale = hd**-0.5
    seq_lens_ext = torch.arange(1, S + 1, device=device, dtype=torch.int32)
    max_blocks = 8
    block_tables_ext = torch.zeros(S, max_blocks, device=device, dtype=torch.int32)
    block_tables_ext[:, 0] = 0

    attn_out = torch.empty((S, nh, hd), device=device, dtype=torch.float16)
    max_ctx = 4096
    paged_attention_v1(
        attn_out,
        q.contiguous(),
        k_cache,
        v_cache,
        nh,
        scale,
        block_tables_ext,
        seq_lens_ext,
        block_size,
        max_ctx,
        None,
        "auto",
        None,
        None,
        num_kv_heads=nkv,
    )

    # Reference: expand k,v for GQA (repeat_interleave per head group)
    k_exp = k.unsqueeze(2).expand(-1, -1, group, -1).reshape(S, nh, hd)
    v_exp = v.unsqueeze(2).expand(-1, -1, group, -1).reshape(S, nh, hd)
    ref = torch.zeros_like(q)
    for i in range(S):
        sl = int(seq_lens_ext[i].item())
        qi = q[i].unsqueeze(0)  # (1, nh, hd)
        K = k_exp[:sl]  # (sl, nh, hd)
        V = v_exp[:sl]
        # scores[h, s] = scale * dot(qi[0,h], K[s,h])
        qi_h = qi.squeeze(0)  # (nh, hd)
        Kh = K.permute(1, 0, 2)  # (nh, sl, hd)
        scores = torch.bmm(qi_h.unsqueeze(1), Kh.transpose(1, 2)).squeeze(1) * scale  # (nh, sl)
        weights = F.softmax(scores, dim=-1)
        Vh = V.permute(1, 0, 2)  # (nh, sl, hd)
        ref[i] = (weights.unsqueeze(-1) * Vh).sum(dim=1)

    assert torch.allclose(attn_out.float(), ref.float(), rtol=1e-2, atol=5e-2)
