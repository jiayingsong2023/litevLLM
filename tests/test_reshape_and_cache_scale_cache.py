# SPDX-License-Identifier: Apache-2.0
"""Step 5 guardrail: ensure ``reshape_and_cache`` does NOT allocate a
fresh 1-element scale tensor per call on the decode hot path.

Before Step 5 the fallback path in
``vllm/kernels/triton/reshape_and_cache.py`` issued
``torch.tensor([k_scale], device=..., dtype=fp32)`` every call, which
showed up in ROCm profiles as a 4-byte ``hipMemcpyWithStream`` H->D
blocking transfer. For Gemma4-31B decode (60 layers x 2 scales/layer)
this was ~120 syncs/step and ~79% of wall time.

This test file pins the new behaviour:

  * Python-scalar k_scale/v_scale resolve to a *cached* 1-element tensor;
    the second call with the same (device, dtype, value) MUST reuse the
    same tensor object.
  * Different values / dtypes / devices get independent cache entries.
  * Tensor inputs bypass the cache (legacy contract preserved).
  * Numerical output of a bf16 (non-fp8, non-int4) write is identical to
    a reference PyTorch implementation.
  * Under a ``torch.tensor`` monkey-patch we observe that after a single
    warmup call, subsequent calls with the same Python-scalar arguments
    do NOT invoke ``torch.tensor([...], device='cuda', ...)`` again.

The last check is the real regression guard for Step 5 because it
measures what the Chrome trace was measuring: per-call small-payload
H->D copies issued inside reshape_and_cache's Python wrapper.
"""
from __future__ import annotations

from typing import Optional

import pytest
import torch

from vllm.kernels.triton import reshape_and_cache as rac_mod
from vllm.kernels.triton.reshape_and_cache import (
    _SCALE_TENSOR_CACHE,
    _clear_scale_tensor_cache,
    _get_scalar_scale_tensor,
    reshape_and_cache,
)


def _has_cuda_or_rocm() -> bool:
    return torch.cuda.is_available()


REQUIRES_GPU = pytest.mark.skipif(
    not _has_cuda_or_rocm(),
    reason="reshape_and_cache Triton kernel requires a CUDA/ROCm device",
)


# ---------------------------------------------------------------------------
# Helpers: build a miniature KV cache that matches kernel layout.
# ---------------------------------------------------------------------------
def _make_kv_caches(
    *,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (num_blocks, block_size, num_kv_heads, head_dim)
    kc = torch.zeros(shape, dtype=dtype, device=device)
    vc = torch.zeros(shape, dtype=dtype, device=device)
    return kc, vc


def _reference_write_bf16(
    k: torch.Tensor, v: torch.Tensor,
    k_cache: torch.Tensor, v_cache: torch.Tensor,
    slot_mapping: torch.Tensor, block_size: int,
) -> None:
    """Plain pytorch equivalent of the non-fp8, non-int4 write path."""
    for t in range(k.shape[0]):
        slot = int(slot_mapping[t].item())
        if slot < 0:
            continue
        blk = slot // block_size
        off = slot % block_size
        k_cache[blk, off] = k[t].to(k_cache.dtype)
        v_cache[blk, off] = v[t].to(v_cache.dtype)


# ---------------------------------------------------------------------------
# 1. Cache behaviour (pure Python, runs without a GPU).
# ---------------------------------------------------------------------------
def test_scalar_scale_cache_dedups_by_value_dtype_and_device(monkeypatch) -> None:
    _clear_scale_tensor_cache()
    # Count how many raw torch.tensor([...]) allocations we make so the
    # test stays meaningful even on CPU-only runners where reshape_and_cache
    # itself cannot be called end-to-end.
    orig_tensor = torch.tensor
    alloc_calls: list = []

    def counting_tensor(*args, **kwargs):
        alloc_calls.append((args, kwargs))
        return orig_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", counting_tensor)

    dev = torch.device("cpu")
    t1 = _get_scalar_scale_tensor(1.0, dev)
    t2 = _get_scalar_scale_tensor(1.0, dev)
    assert t1 is t2, "same (device, dtype, value) must reuse tensor"
    assert len(alloc_calls) == 1, f"expected exactly one allocation, got {len(alloc_calls)}"

    # Different scalar value -> new cache entry.
    t3 = _get_scalar_scale_tensor(0.5, dev)
    assert t3 is not t1
    assert len(alloc_calls) == 2

    # Different dtype -> new cache entry.
    t4 = _get_scalar_scale_tensor(1.0, dev, dtype=torch.float16)
    assert t4 is not t1
    assert len(alloc_calls) == 3

    # Sanity: stored values match inputs exactly.
    assert float(t1.item()) == 1.0
    assert float(t3.item()) == 0.5
    assert float(t4.item()) == 1.0


def test_scalar_scale_cache_accepts_str_or_device(monkeypatch) -> None:
    _clear_scale_tensor_cache()
    orig_tensor = torch.tensor
    alloc_calls: list = []

    def counting_tensor(*args, **kwargs):
        alloc_calls.append((args, kwargs))
        return orig_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", counting_tensor)

    # Pass device as a string; implementation should normalize to
    # torch.device before hashing so str and torch.device("cpu") dedupe.
    t_str = _get_scalar_scale_tensor(1.0, "cpu")
    t_dev = _get_scalar_scale_tensor(1.0, torch.device("cpu"))
    assert t_str is t_dev
    assert len(alloc_calls) == 1


# ---------------------------------------------------------------------------
# 2. End-to-end GPU: cache reuse under the real reshape_and_cache wrapper.
# ---------------------------------------------------------------------------
@REQUIRES_GPU
def test_reshape_and_cache_does_not_realloc_scale_after_warmup(monkeypatch) -> None:
    _clear_scale_tensor_cache()
    device = torch.device("cuda")
    num_blocks, block_size = 4, 16
    num_kv_heads, head_dim = 2, 64
    num_tokens = 1

    kc, vc = _make_kv_caches(
        num_blocks=num_blocks, block_size=block_size,
        num_kv_heads=num_kv_heads, head_dim=head_dim,
        dtype=torch.bfloat16, device=device,
    )
    k = torch.randn(num_tokens, num_kv_heads, head_dim,
                    dtype=torch.bfloat16, device=device)
    v = torch.randn_like(k)
    slot_mapping = torch.tensor([3], dtype=torch.int32, device=device)

    # Warmup call: first time we hit this (device, dtype, value) tuple,
    # cache miss is expected.
    reshape_and_cache(k, v, kc, vc, slot_mapping, "bf16", 1.0, 1.0)
    torch.cuda.synchronize()

    # The production path calls ``_get_scalar_scale_tensor`` with
    # ``key.device`` which carries the real CUDA index (cuda:0), not the
    # generic torch.device('cuda'). We look the entry up via the tensor's
    # device to match that contract.
    cache_key = (k.device, torch.float32, 1.0)
    assert cache_key in _SCALE_TENSOR_CACHE, (
        f"expected cache miss on first call to populate the entry, "
        f"got cache={list(_SCALE_TENSOR_CACHE.keys())}"
    )
    cached_tensor = _SCALE_TENSOR_CACHE[cache_key]

    # Now patch torch.tensor so any hot-path allocation would show up.
    orig_tensor = torch.tensor
    hot_path_allocs: list = []

    def trap_tensor(*args, **kwargs):
        hot_path_allocs.append((args, kwargs))
        return orig_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", trap_tensor)

    for _ in range(20):
        reshape_and_cache(k, v, kc, vc, slot_mapping, "bf16", 1.0, 1.0)
    torch.cuda.synchronize()

    assert hot_path_allocs == [], (
        f"expected zero torch.tensor() calls after warmup, saw "
        f"{len(hot_path_allocs)}: {hot_path_allocs[:3]}"
    )
    assert _SCALE_TENSOR_CACHE[cache_key] is cached_tensor, (
        "cached scale tensor identity must not change across calls"
    )


@REQUIRES_GPU
def test_reshape_and_cache_tensor_input_bypasses_cache() -> None:
    _clear_scale_tensor_cache()
    device = torch.device("cuda")
    block_size = 16
    kc, vc = _make_kv_caches(
        num_blocks=4, block_size=block_size,
        num_kv_heads=2, head_dim=64,
        dtype=torch.bfloat16, device=device,
    )
    k = torch.randn(1, 2, 64, dtype=torch.bfloat16, device=device)
    v = torch.randn_like(k)
    slot_mapping = torch.tensor([1], dtype=torch.int32, device=device)

    k_scale_t = torch.tensor([0.75], dtype=torch.float32, device=device)
    v_scale_t = torch.tensor([0.25], dtype=torch.float32, device=device)
    # Legacy caller pattern: scales already live on device as tensors.
    reshape_and_cache(k, v, kc, vc, slot_mapping, "bf16", k_scale_t, v_scale_t)
    torch.cuda.synchronize()

    # When scales come in as tensors, we MUST NOT populate the scalar
    # cache (that cache is keyed by python float).
    assert _SCALE_TENSOR_CACHE == {}, (
        "tensor-valued scales must bypass the scalar cache entirely"
    )


# ---------------------------------------------------------------------------
# 3. Correctness on the bf16 "non-fp8, non-int4" path.
# ---------------------------------------------------------------------------
@REQUIRES_GPU
def test_reshape_and_cache_bf16_matches_reference() -> None:
    _clear_scale_tensor_cache()
    device = torch.device("cuda")
    num_blocks, block_size = 8, 16
    num_kv_heads, head_dim = 2, 64
    num_tokens = 5

    kc, vc = _make_kv_caches(
        num_blocks=num_blocks, block_size=block_size,
        num_kv_heads=num_kv_heads, head_dim=head_dim,
        dtype=torch.bfloat16, device=device,
    )
    kc_ref = kc.clone()
    vc_ref = vc.clone()

    k = torch.randn(num_tokens, num_kv_heads, head_dim,
                    dtype=torch.bfloat16, device=device)
    v = torch.randn_like(k)
    # Mix valid slots with one -1 to verify the skip branch still holds.
    slot_mapping = torch.tensor([5, -1, 20, 35, 3],
                                dtype=torch.int32, device=device)

    reshape_and_cache(k, v, kc, vc, slot_mapping, "bf16", 1.0, 1.0)
    torch.cuda.synchronize()

    _reference_write_bf16(k, v, kc_ref, vc_ref, slot_mapping, block_size)

    assert torch.equal(kc, kc_ref), "bf16 KV write must be bit-identical to reference"
    assert torch.equal(vc, vc_ref), "bf16 VV write must be bit-identical to reference"


# ---------------------------------------------------------------------------
# 4. Boundary cases (0-token and full block fill) - safety net.
# ---------------------------------------------------------------------------
@REQUIRES_GPU
def test_reshape_and_cache_zero_tokens_is_noop() -> None:
    _clear_scale_tensor_cache()
    device = torch.device("cuda")
    block_size = 16
    kc, vc = _make_kv_caches(
        num_blocks=2, block_size=block_size,
        num_kv_heads=2, head_dim=64,
        dtype=torch.bfloat16, device=device,
    )
    kc_before = kc.clone()
    vc_before = vc.clone()
    k = torch.empty((0, 2, 64), dtype=torch.bfloat16, device=device)
    v = torch.empty_like(k)
    slot_mapping = torch.empty((0,), dtype=torch.int32, device=device)

    reshape_and_cache(k, v, kc, vc, slot_mapping, "bf16", 1.0, 1.0)
    torch.cuda.synchronize()

    assert torch.equal(kc, kc_before)
    assert torch.equal(vc, vc_before)


@REQUIRES_GPU
def test_reshape_and_cache_module_cache_survives_across_invocations() -> None:
    """Across two logically separate calls with the same scalar scale,
    the cached scale tensor must not be rebuilt, which is the whole point
    of Step 5. This is a thinner variant of the warmup test above that
    pins the identity check without a monkeypatch.
    """
    _clear_scale_tensor_cache()
    device = torch.device("cuda")
    block_size = 16
    kc, vc = _make_kv_caches(
        num_blocks=2, block_size=block_size,
        num_kv_heads=2, head_dim=64,
        dtype=torch.bfloat16, device=device,
    )
    k = torch.randn(1, 2, 64, dtype=torch.bfloat16, device=device)
    v = torch.randn_like(k)
    slot_mapping = torch.tensor([0], dtype=torch.int32, device=device)

    reshape_and_cache(k, v, kc, vc, slot_mapping, "bf16", 1.0, 1.0)
    torch.cuda.synchronize()
    first = _SCALE_TENSOR_CACHE[(k.device, torch.float32, 1.0)]
    first_ptr = first.data_ptr()

    reshape_and_cache(k, v, kc, vc, slot_mapping, "bf16", 1.0, 1.0)
    torch.cuda.synchronize()
    second = _SCALE_TENSOR_CACHE[(k.device, torch.float32, 1.0)]

    assert second is first
    assert second.data_ptr() == first_ptr
