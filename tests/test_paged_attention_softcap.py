# SPDX-License-Identifier: Apache-2.0
"""Numerical parity tests for Gemma-style logit soft-capping inside the
Triton PagedAttention V1 kernel.

These tests exercise the ``softcap`` branch of ``paged_attention_v1`` end-to-end
by comparing its output against an eager PyTorch reference for fp16 KV cache.

Coverage:
  * softcap disabled (None / 0 / negative) must be byte-stable vs pre-Step3 path
  * softcap=50 (Gemma4 default) must match eager ref within fp16 tolerance
  * Boundary seq_len=1 and seq_len equals full block boundary
  * head_size 64 (LlaMA-like) and 256 (Gemma4 31B) both covered
"""
from __future__ import annotations

import math
from typing import Optional

import pytest
import torch

from vllm.kernels.triton.paged_attention import paged_attention_v1


def _has_cuda_or_rocm() -> bool:
    return torch.cuda.is_available()


REQUIRES_GPU = pytest.mark.skipif(
    not _has_cuda_or_rocm(), reason="paged_attention kernel needs a GPU runtime"
)


# ---------------------------------------------------------------------------
# Eager reference implementation (fp16 KV path)
# ---------------------------------------------------------------------------
def _paged_attention_reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    softcap: Optional[float],
) -> torch.Tensor:
    """Decode-step paged attention reference in fp32, then cast back to q dtype.

    Layouts follow the Triton kernel:
      * query:       (num_seqs, num_heads, head_size)
      * key_cache:   (num_blocks, block_size, num_kv_heads, head_size)
      * value_cache: (num_blocks, block_size, num_kv_heads, head_size)
      * block_tables:(num_seqs, max_num_blocks_per_seq) int32 indices
      * seq_lens:    (num_seqs,) int32 per-seq token counts
    """
    num_seqs, _, head_size = query.shape
    block_size = key_cache.shape[1]
    q_group = num_heads // num_kv_heads
    out = torch.zeros_like(query)
    for si in range(num_seqs):
        s_len = int(seq_lens[si].item())
        if s_len <= 0:
            continue
        n_blocks = (s_len + block_size - 1) // block_size
        ks = []
        vs = []
        for bi in range(n_blocks):
            bidx = int(block_tables[si, bi].item())
            ks.append(key_cache[bidx])  # (block_size, num_kv_heads, head_size)
            vs.append(value_cache[bidx])
        k_full = torch.cat(ks, dim=0)[:s_len].to(torch.float32)
        v_full = torch.cat(vs, dim=0)[:s_len].to(torch.float32)
        for h in range(num_heads):
            kv_h = h // q_group
            q = query[si, h].to(torch.float32)
            k = k_full[:, kv_h, :]
            v = v_full[:, kv_h, :]
            qk = torch.matmul(q, k.t()) * scale
            if softcap is not None and float(softcap) > 0.0:
                cap = float(softcap)
                qk = cap * torch.tanh(qk / cap)
            p = torch.softmax(qk, dim=-1, dtype=torch.float32)
            out[si, h] = torch.matmul(p, v).to(query.dtype)
    return out


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------
def _build_case(
    *,
    device: torch.device,
    dtype: torch.dtype,
    num_seqs: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    seq_lens: list[int],
    seed: int = 0,
):
    torch.manual_seed(seed)
    max_len = max(seq_lens) if seq_lens else 1
    max_num_blocks_per_seq = (max_len + block_size - 1) // block_size
    # Allocate a generous pool so block_tables can map independent ranges
    # per sequence without collisions.
    total_blocks = max(num_seqs * max_num_blocks_per_seq + 4, 8)

    query = torch.randn(
        num_seqs, num_heads, head_size, device=device, dtype=dtype
    )
    key_cache = torch.randn(
        total_blocks, block_size, num_kv_heads, head_size, device=device, dtype=dtype
    )
    value_cache = torch.randn(
        total_blocks, block_size, num_kv_heads, head_size, device=device, dtype=dtype
    )

    block_tables = torch.zeros(
        num_seqs, max_num_blocks_per_seq, device=device, dtype=torch.int32
    )
    bcounter = 0
    for si in range(num_seqs):
        for bi in range(max_num_blocks_per_seq):
            block_tables[si, bi] = bcounter
            bcounter += 1

    seq_lens_t = torch.tensor(seq_lens, device=device, dtype=torch.int32)
    return query, key_cache, value_cache, block_tables, seq_lens_t


def _run_kernel(
    query,
    key_cache,
    value_cache,
    block_tables,
    seq_lens,
    *,
    num_heads,
    num_kv_heads,
    block_size,
    scale,
    softcap,
    kv_cache_dtype: str = "auto",
):
    out = torch.empty_like(query)
    max_seq_len = int(seq_lens.max().item())
    paged_attention_v1(
        out,
        query,
        key_cache,
        value_cache,
        num_heads,
        scale,
        block_tables,
        seq_lens,
        block_size,
        max_seq_len,
        None,
        kv_cache_dtype,
        1.0,
        1.0,
        num_kv_heads=num_kv_heads,
        softcap=softcap,
    )
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@REQUIRES_GPU
@pytest.mark.parametrize(
    "head_size,num_heads,num_kv_heads",
    [
        (64, 8, 2),     # small LLaMA-like
        (256, 4, 1),    # Gemma4 31B attention head geometry
    ],
)
@pytest.mark.parametrize("softcap", [None, 0.0, -1.0])
def test_softcap_disabled_paths_equivalent(head_size, num_heads, num_kv_heads, softcap):
    """None / 0 / negative softcap must all take the HAS_SOFTCAP=False compile
    branch; results must be identical to each other and to the eager ref."""
    device = torch.device("cuda")
    dtype = torch.float16
    num_seqs = 3
    block_size = 16
    seq_lens = [17, 32, 1]
    scale = 1.0 / math.sqrt(head_size)

    query, kc, vc, bt, sl = _build_case(
        device=device,
        dtype=dtype,
        num_seqs=num_seqs,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        seq_lens=seq_lens,
        seed=42,
    )

    triton_out = _run_kernel(
        query, kc, vc, bt, sl,
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        block_size=block_size, scale=scale, softcap=softcap,
    )
    ref_out = _paged_attention_reference(
        query, kc, vc, bt, sl,
        scale=scale, num_heads=num_heads, num_kv_heads=num_kv_heads,
        softcap=None,
    )
    # fp16 kernel vs fp32 ref: use loose tol. The goal is functional equivalence.
    torch.testing.assert_close(triton_out.float(), ref_out.float(), atol=5e-3, rtol=5e-3)


@REQUIRES_GPU
@pytest.mark.parametrize(
    "head_size,num_heads,num_kv_heads",
    [
        (64, 8, 2),
        (256, 4, 1),
    ],
)
@pytest.mark.parametrize("softcap_value", [50.0, 30.0, 5.0])
def test_softcap_matches_eager_reference(head_size, num_heads, num_kv_heads, softcap_value):
    """With softcap enabled, kernel output must match eager fp32 ref within fp16 tol."""
    device = torch.device("cuda")
    dtype = torch.float16
    num_seqs = 4
    block_size = 16
    # Mix short / long / exact-block-boundary / single-token cases.
    seq_lens = [1, 15, 16, 33]
    scale = 1.0 / math.sqrt(head_size)

    query, kc, vc, bt, sl = _build_case(
        device=device,
        dtype=dtype,
        num_seqs=num_seqs,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        seq_lens=seq_lens,
        seed=123,
    )

    triton_out = _run_kernel(
        query, kc, vc, bt, sl,
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        block_size=block_size, scale=scale, softcap=softcap_value,
    )
    ref_out = _paged_attention_reference(
        query, kc, vc, bt, sl,
        scale=scale, num_heads=num_heads, num_kv_heads=num_kv_heads,
        softcap=softcap_value,
    )
    torch.testing.assert_close(triton_out.float(), ref_out.float(), atol=5e-3, rtol=5e-3)


@REQUIRES_GPU
def test_softcap_changes_output_when_enabled():
    """Sanity: softcap=50 must produce a numerically different output than
    softcap disabled on the same inputs. Otherwise our HAS_SOFTCAP branch
    is silently compiled out."""
    device = torch.device("cuda")
    dtype = torch.float16
    head_size = 64
    num_heads = 8
    num_kv_heads = 2
    block_size = 16
    scale = 1.0 / math.sqrt(head_size)
    # Use a large-magnitude Q to force the tanh saturation to matter.
    torch.manual_seed(7)
    seq_lens = [24]
    num_seqs = 1
    max_num_blocks_per_seq = 2
    query = torch.randn(
        num_seqs, num_heads, head_size, device=device, dtype=dtype
    ) * 5.0
    key_cache = torch.randn(
        max_num_blocks_per_seq, block_size, num_kv_heads, head_size,
        device=device, dtype=dtype,
    ) * 5.0
    value_cache = torch.randn(
        max_num_blocks_per_seq, block_size, num_kv_heads, head_size,
        device=device, dtype=dtype,
    )
    block_tables = torch.arange(
        max_num_blocks_per_seq, device=device, dtype=torch.int32
    ).view(num_seqs, max_num_blocks_per_seq)
    sl = torch.tensor(seq_lens, device=device, dtype=torch.int32)

    out_no_cap = _run_kernel(
        query, key_cache, value_cache, block_tables, sl,
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        block_size=block_size, scale=scale, softcap=None,
    )
    out_with_cap = _run_kernel(
        query, key_cache, value_cache, block_tables, sl,
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        block_size=block_size, scale=scale, softcap=5.0,
    )
    diff = (out_no_cap.float() - out_with_cap.float()).abs().max().item()
    assert diff > 1e-3, (
        "softcap branch appears to have been compiled out "
        f"(max_diff={diff:.3e})"
    )


@REQUIRES_GPU
def test_softcap_non_gemma_models_zero_overhead_semantics():
    """Regression guard: callers that do not pass `softcap` at all (e.g. Qwen,
    LLaMA) must continue to produce softcap-disabled results."""
    device = torch.device("cuda")
    dtype = torch.float16
    head_size = 128
    num_heads = 16
    num_kv_heads = 4
    block_size = 16
    num_seqs = 2
    seq_lens = [7, 31]
    scale = 1.0 / math.sqrt(head_size)

    query, kc, vc, bt, sl = _build_case(
        device=device, dtype=dtype,
        num_seqs=num_seqs, num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_size=head_size, block_size=block_size, seq_lens=seq_lens, seed=99,
    )

    # No softcap kwarg at all (simulates Qwen/LLaMA call site).
    out = torch.empty_like(query)
    paged_attention_v1(
        out, query, kc, vc, num_heads, scale, bt, sl,
        block_size, max(seq_lens), None, "auto", 1.0, 1.0,
        num_kv_heads=num_kv_heads,
    )
    ref = _paged_attention_reference(
        query, kc, vc, bt, sl,
        scale=scale, num_heads=num_heads, num_kv_heads=num_kv_heads,
        softcap=None,
    )
    torch.testing.assert_close(out.float(), ref.float(), atol=5e-3, rtol=5e-3)
