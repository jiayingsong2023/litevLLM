# SPDX-License-Identifier: Apache-2.0
"""Measure the Gemma4 12B M=1 attention front half with production tensors.

This is intentionally a diagnostic, not a runtime switch.  It uses layer zero
from the real checkpoint, the production fused-QKV dispatch, position 512, and
the engine's allocated FP8 KV cache.  A fusion candidate is only justified if
the measured post-QKV work is large enough to repay a new exact kernel.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "models/gemma-4-12B-it-AWQ-INT4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--context-tokens", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _median_ms(fn: Callable[[], Any], repetitions: int) -> float:
    import torch

    fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repetitions):
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        started.record()
        fn()
        finished.record()
        finished.synchronize()
        samples.append(float(started.elapsed_time(finished)))
    return float(statistics.median(samples))


def main() -> None:
    args = parse_args()
    if args.context_tokens < 1:
        raise ValueError("--context-tokens must be positive")
    if args.repetitions < 1:
        raise ValueError("--repetitions must be positive")

    import torch

    from vllm import LLM
    from vllm.kernels.triton.reshape_and_cache import reshape_and_cache
    from vllm.model_executor.models.gemma4.attention import _try_fused_awq_qkv_decode

    llm = LLM(
        args.model,
        max_num_seqs=1,
        max_model_len=args.context_tokens + 16,
    )
    attn = llm.model.model.layers[0].self_attn
    inf_config = llm.engine.inf_config
    x = torch.randn(
        (1, 1, int(attn.q_proj.input_size)),
        device="cuda",
        dtype=torch.bfloat16,
    )
    position = torch.tensor([args.context_tokens], device="cuda", dtype=torch.long)
    fused = _try_fused_awq_qkv_decode(
        x,
        attn.q_proj,
        attn.k_proj,
        attn.v_proj,
        inf_config=inf_config,
    )
    if fused is None:
        raise RuntimeError("production fused M=1 QKV dispatch was unavailable")
    q_raw, k_raw, v_raw = fused
    bsz, seqlen = x.shape[:2]
    q_raw = q_raw.view(bsz, seqlen, attn.num_heads, attn.head_dim)
    k_raw = k_raw.view(bsz, seqlen, attn.num_kv_heads, attn.head_dim)
    v_raw = v_raw.view(bsz, seqlen, attn.num_kv_heads, attn.head_dim)
    # Build and warm the production RoPE cache before collecting timings.
    q_norm = attn._apply_head_norm(attn.q_norm, q_raw)
    k_norm = attn._apply_head_norm(attn.k_norm, k_raw)
    q_rope, k_rope = attn.rotary_emb(
        position,
        q_norm,
        k_norm,
        max_position_plus_one_cpu=args.context_tokens + 1,
        inf_config=inf_config,
    )
    v_norm = attn._apply_head_norm_noscale(v_raw, attn.v_norm_eps)

    k_cache, v_cache = llm.engine.kv_caches[0]
    block_size = int(k_cache.shape[1])
    slot_mapping = torch.tensor([args.context_tokens], device="cuda", dtype=torch.long)
    if int(slot_mapping.item()) >= int(k_cache.shape[0]) * block_size:
        raise RuntimeError("audit position exceeds allocated KV cache")
    kv_scale_cache = llm.engine.kv_scale_caches[0]
    if kv_scale_cache is None:
        k_scale_cache, v_scale_cache = (None, None)
    else:
        k_scale_cache, v_scale_cache = kv_scale_cache

    def write_cache() -> None:
        reshape_and_cache(
            k_rope.reshape(-1, attn.num_kv_heads, attn.head_dim).contiguous(),
            v_norm.reshape(-1, attn.num_kv_heads, attn.head_dim).contiguous(),
            k_cache,
            v_cache,
            slot_mapping,
            inf_config.kv_type,
            inf_config.k_scale,
            inf_config.v_scale,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
        )

    def complete_postprocess() -> None:
        q = attn._apply_head_norm(attn.q_norm, q_raw)
        k = attn._apply_head_norm(attn.k_norm, k_raw)
        v = attn._apply_head_norm_noscale(v_raw, attn.v_norm_eps)
        _q, k = attn.rotary_emb(
            position,
            q,
            k,
            max_position_plus_one_cpu=args.context_tokens + 1,
            inf_config=inf_config,
        )
        reshape_and_cache(
            k.reshape(-1, attn.num_kv_heads, attn.head_dim).contiguous(),
            v.reshape(-1, attn.num_kv_heads, attn.head_dim).contiguous(),
            k_cache,
            v_cache,
            slot_mapping,
            inf_config.kv_type,
            inf_config.k_scale,
            inf_config.v_scale,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
        )

    timings_ms = {
        "fused_qkv": _median_ms(
            lambda: _try_fused_awq_qkv_decode(
                x,
                attn.q_proj,
                attn.k_proj,
                attn.v_proj,
                inf_config=inf_config,
            ),
            args.repetitions,
        ),
        "q_norm": _median_ms(
            lambda: attn._apply_head_norm(attn.q_norm, q_raw), args.repetitions
        ),
        "k_norm": _median_ms(
            lambda: attn._apply_head_norm(attn.k_norm, k_raw), args.repetitions
        ),
        "v_norm": _median_ms(
            lambda: attn._apply_head_norm_noscale(v_raw, attn.v_norm_eps),
            args.repetitions,
        ),
        "q_and_k_rope": _median_ms(
            lambda: attn.rotary_emb(
                position,
                q_norm,
                k_norm,
                max_position_plus_one_cpu=args.context_tokens + 1,
                inf_config=inf_config,
            ),
            args.repetitions,
        ),
        "kv_write": _median_ms(write_cache, args.repetitions),
        "postprocess_total": _median_ms(complete_postprocess, args.repetitions),
    }
    result = {
        "model": args.model,
        "context_tokens": args.context_tokens,
        "repetitions": args.repetitions,
        "layer": 0,
        "q_shape": list(q_raw.shape),
        "k_shape": list(k_raw.shape),
        "v_shape": list(v_raw.shape),
        "kv_cache_dtype": str(k_cache.dtype),
        "timings_ms": timings_ms,
        "postprocess_share_of_front_half": timings_ms["postprocess_total"]
        / (timings_ms["fused_qkv"] + timings_ms["postprocess_total"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
