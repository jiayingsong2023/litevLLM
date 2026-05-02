# SPDX-License-Identifier: Apache-2.0
"""Profiling microbenchmarks for reshape_and_cache and paged_attention.

Usage:
    uv run python tests/tools/profile_kernel_registers.py
"""

from __future__ import annotations

import time
from typing import Any

import torch

from vllm.kernels.triton.paged_attention import (
    _paged_attention_kernel,
    _select_paged_attention_launch_config,
)
from vllm.kernels.triton.reshape_and_cache import reshape_and_cache


def _fmt_us(seconds: float) -> str:
    return f"{seconds * 1e6:.1f} us"


def _fmt_bw(total_bytes: int, seconds: float) -> str:
    bw = total_bytes / seconds / 1e9
    return f"{bw:.1f} GB/s"


# ── reshape_and_cache profiling ────────────────────────────────────────────

def _profile_reshape_and_cache() -> None:
    print("=" * 72)
    print("reshape_and_cache — register-pressure profiling")
    print("=" * 72)

    device = torch.device("cuda:0")
    head_dim = 256
    num_kv_heads = 8
    num_tokens = 16  # one block worth
    block_size = 16
    sig_dim = 32
    num_total_blocks = 1024
    warmup = 20
    iters = 200

    key = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=torch.float16, device=device)
    value = key.clone()
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    torch.cuda.synchronize()

    configs = [
        {
            "label": "fp16 (lightest)",
            "kv_cache_dtype": "auto",
            "k_scale": 1.0,
            "v_scale": 1.0,
            "k_scale_cache": None,
            "v_scale_cache": None,
            "sig_temp": None,
        },
        {
            "label": "int4 (no dynamic scale, no sig)",
            "kv_cache_dtype": "turbo_int4",
            "k_scale": 1.0,
            "v_scale": 1.0,
            "k_scale_cache": None,
            "v_scale_cache": None,
            "sig_temp": None,
        },
        {
            "label": "int4 + dynamic scale (no sig)",
            "kv_cache_dtype": "turbo_int4",
            "k_scale": 1.0,
            "v_scale": 1.0,
            "k_scale_cache": torch.randn(
                num_total_blocks, block_size, num_kv_heads, dtype=torch.float32, device=device
            ),
            "v_scale_cache": torch.randn(
                num_total_blocks, block_size, num_kv_heads, dtype=torch.float32, device=device
            ),
            "sig_temp": None,
        },
        {
            "label": "int4 + dynamic scale + WRITE_SIG (heaviest)",
            "kv_cache_dtype": "turbo_int4",
            "k_scale": 1.0,
            "v_scale": 1.0,
            "k_scale_cache": torch.randn(
                num_total_blocks, block_size, num_kv_heads, dtype=torch.float32, device=device
            ),
            "v_scale_cache": torch.randn(
                num_total_blocks, block_size, num_kv_heads, dtype=torch.float32, device=device
            ),
            "sig_temp": torch.zeros(
                num_total_blocks, block_size, num_kv_heads, sig_dim,
                dtype=torch.float16, device=device,
            ),
        },
    ]

    for cfg in configs:
        key_cache_shape = (
            num_total_blocks, block_size, num_kv_heads, head_dim
        )
        if cfg["kv_cache_dtype"] == "turbo_int4":
            key_cache_shape = (num_total_blocks, block_size, num_kv_heads, head_dim // 2)
            kv_dtype: Any = torch.uint8
        elif cfg["kv_cache_dtype"] == "fp8":
            kv_dtype = torch.float8_e4m3fnuz
        else:
            kv_dtype = torch.float16

        key_cache = torch.zeros(key_cache_shape, dtype=kv_dtype, device=device)
        value_cache = torch.zeros(key_cache_shape, dtype=kv_dtype, device=device)

        fn = lambda: reshape_and_cache(  # noqa: E731
            key, value, key_cache, value_cache, slot_mapping,
            cfg["kv_cache_dtype"],
            k_scale=cfg["k_scale"], v_scale=cfg["v_scale"],
            k_scale_cache=cfg["k_scale_cache"],
            v_scale_cache=cfg["v_scale_cache"],
            sig_temp=cfg["sig_temp"], sig_dim=sig_dim,
        )

        # warmup
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        for _ in range(iters):
            fn()
        end_ev.record()
        torch.cuda.synchronize()
        elapsed_s = start_ev.elapsed_time(end_ev) / 1e3  # ms -> s
        elapsed_per = elapsed_s / iters

        # Approximate I/O: read K+V (2 * tokens * heads * head_dim * 2 bytes)
        # plus write to cache (~ same or half for int4)
        io_bytes = 2 * num_tokens * num_kv_heads * head_dim * 2  # read K, V
        if cfg["kv_cache_dtype"] == "turbo_int4":
            io_bytes += 2 * num_tokens * num_kv_heads * (head_dim // 2)  # write kv cache
        elif cfg["kv_cache_dtype"] == "fp8":
            io_bytes += 2 * num_tokens * num_kv_heads * head_dim  # write kv cache
        else:
            io_bytes += 2 * num_tokens * num_kv_heads * head_dim * 2  # write kv cache

        print(
            f"  {cfg['label']:45s}  {_fmt_us(elapsed_per):>12s}  {_fmt_bw(io_bytes, elapsed_per):>10s}"  # noqa: E501
        )


# ── paged_attention profiling ──────────────────────────────────────────────

def _profile_paged_attention() -> None:
    print()
    print("=" * 72)
    print("paged_attention — launch config profiling (head_size=256)")
    print("=" * 72)

    device = torch.device("cuda:0")
    head_size = 256
    num_kv_heads = 8
    num_heads = 32
    block_size = 16
    max_blocks_per_seq = 512
    num_total_blocks = 2048
    warmup = 10
    iters = 100

    key_cache = torch.randn(
        num_total_blocks, block_size, num_kv_heads, head_size,
        dtype=torch.float16, device=device,
    )
    value_cache = key_cache.clone()

    launch_configs = [
        (2, 2, "w2_s2"),
        (4, 1, "w4_s1"),
        (4, 2, "w4_s2"),
    ]

    for num_seqs in [1, 2, 4]:
        seq_lens = [min(64, (i + 1) * 16) for i in range(num_seqs)]
        num_blocks = max(seq_lens) // block_size + (1 if max(seq_lens) % block_size else 0)
        num_blocks = max(num_blocks, 1)

        block_tables = torch.zeros(num_seqs, max_blocks_per_seq, dtype=torch.int32, device=device)
        for s in range(num_seqs):
            for b in range(num_blocks):
                block_tables[s, b] = s * max_blocks_per_seq + b

        seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        query = torch.randn(num_seqs, num_heads, head_size, dtype=torch.float16, device=device)
        output = torch.zeros(num_seqs, num_heads, head_size, dtype=torch.float16, device=device)

        default_config = _select_paged_attention_launch_config(
            num_seqs=num_seqs, head_size=head_size, block_size=block_size,
            is_int4=False, is_fp8=False,
        )
        print(f"\n  num_seqs={num_seqs}  (default: w{default_config[0]}_s{default_config[1]})")

        s_k = key_cache.stride()
        s_v = value_cache.stride()

        for num_warps, num_stages, label in launch_configs:
            grid = (num_seqs * num_heads,)
            torch.cuda.synchronize()

            # warmup
            for _ in range(warmup):
                _paged_attention_kernel[grid](
                    output, query, key_cache, value_cache,
                    block_tables, seq_lens_t,
                    output, output,  # dummy K_Ptrs, V_Ptrs (unused)
                    1.0,  # scale
                    num_seqs, num_heads, num_kv_heads,
                    head_size, block_size,
                    max_blocks_per_seq,
                    output.stride(0), output.stride(1), output.stride(2),
                    query.stride(0), query.stride(1), query.stride(2),
                    s_k[0], s_k[1], s_k[2], s_k[3],
                    s_v[0], s_v[1], s_v[2], s_v[3],
                    block_tables.stride(0), block_tables.stride(1),
                    seq_lens_t.stride(0),
                    1.0, 1.0,  # k_scale, v_scale (dummy, not HAS_ROW_SCALE)
                    output, output, 0, 0, 0, 0, 0, 0,  # dummy row scale ptrs
                    0.0,  # softcap_val
                    IS_FP8=False, IS_INT4=False,
                    IS_CACHED=False, HAS_ROW_SCALE=False,
                    HAS_SOFTCAP=False, USE_SELECTION=False,
                    SELECT_STRIDE=1, MIN_SELECTED_BLOCKS=4,
                    BLOCK_D=head_size, BLOCK_N=block_size,
                    num_warps=num_warps, num_stages=num_stages,
                )
            torch.cuda.synchronize()

            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            for _ in range(iters):
                _paged_attention_kernel[grid](
                    output, query, key_cache, value_cache,
                    block_tables, seq_lens_t,
                    output, output,
                    1.0,
                    num_seqs, num_heads, num_kv_heads,
                    head_size, block_size,
                    max_blocks_per_seq,
                    output.stride(0), output.stride(1), output.stride(2),
                    query.stride(0), query.stride(1), query.stride(2),
                    s_k[0], s_k[1], s_k[2], s_k[3],
                    s_v[0], s_v[1], s_v[2], s_v[3],
                    block_tables.stride(0), block_tables.stride(1),
                    seq_lens_t.stride(0),
                    1.0, 1.0,
                    output, output, 0, 0, 0, 0, 0, 0,
                    0.0,
                    IS_FP8=False, IS_INT4=False,
                    IS_CACHED=False, HAS_ROW_SCALE=False,
                    HAS_SOFTCAP=False, USE_SELECTION=False,
                    SELECT_STRIDE=1, MIN_SELECTED_BLOCKS=4,
                    BLOCK_D=head_size, BLOCK_N=block_size,
                    num_warps=num_warps, num_stages=num_stages,
                )
            end_ev.record()
            torch.cuda.synchronize()
            elapsed_s = start_ev.elapsed_time(end_ev) / 1e3
            elapsed_per = elapsed_s / iters

            kv_blocks = num_seqs * num_blocks
            io_bytes = (
                num_seqs * num_heads * head_size * 2
                + num_seqs * num_heads * head_size * 2
                + kv_blocks * block_size * num_kv_heads * head_size * 2
                + kv_blocks * block_size * num_kv_heads * head_size * 2
            )
            marker = " *" if (num_warps, num_stages) == default_config else ""
            print(
                f"    {label:6s}  {_fmt_us(elapsed_per):>12s}  {_fmt_bw(io_bytes, elapsed_per):>10s}{marker}"  # noqa: E501
            )


# ── main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _profile_reshape_and_cache()
    _profile_paged_attention()
