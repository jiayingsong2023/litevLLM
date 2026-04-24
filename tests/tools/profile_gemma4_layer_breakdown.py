#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Profile Gemma4-31B LiteEngine runs with model-native layer spans plus AWQ runtime stats.

This tool is intentionally lightweight:
- it enables FASTINFERENCE_GEMMA4_LAYER_PROFILE so Gemma4's internal spans are dumped
- it measures prefill/decode step wall times
- it prints AWQ fused/cache counters for the same run

Usage:
  uv run python tests/tools/profile_gemma4_layer_breakdown.py \
    --model models/gemma-4-31B-it-AWQ-4bit \
    --prompt-tokens 512 --decode-steps 32
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from vllm.sampling_params import SamplingParams

_ROOT = Path(__file__).resolve().parents[2]


def _load_gemma4_smoke_module() -> Any:
    import importlib.util
    import sys

    module_path = _ROOT / "tests" / "tools" / "gemma4_single_prompt_smoke.py"
    spec = importlib.util.spec_from_file_location("gemma4_single_prompt_smoke", module_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_prompt(tokenizer: Any, target_tokens: int) -> str:
    sentence = (
        "Explain how to improve inference throughput for a memory-bound transformer model "
        "without sacrificing output quality. "
    )
    target_tokens = max(16, int(target_tokens))
    prompt_text = sentence * max(8, target_tokens // 10)
    token_ids = tokenizer.encode(prompt_text)
    while len(token_ids) < target_tokens:
        prompt_text += sentence
        token_ids = tokenizer.encode(prompt_text)
    if len(token_ids) > target_tokens:
        prompt_text = tokenizer.decode(token_ids[:target_tokens])
    return prompt_text


def _prefill_done(engine: Any) -> bool:
    request_ids = list(engine.scheduler.request_ids())
    if not request_ids:
        return True
    return all(
        not bool(engine.scheduler.get_request(rid).get("is_prefill", False))
        for rid in request_ids
    )


def _step_with_timing(engine: Any) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    engine.step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def _mean(xs: list[float]) -> float:
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


def _p50(xs: list[float]) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    return ys[len(ys) // 2]


def _has_running_requests(engine: Any) -> bool:
    return int(getattr(engine, "active_request_count", 0)) > 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma4 LiteEngine layer profiler")
    parser.add_argument("--model", type=str, default="", help="Model path (default: auto-discover local Gemma4 31B)")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup-decode", type=int, default=4)
    parser.add_argument("--decode-steps", type=int, default=24)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--kv-type", type=str, default="turbo_int4")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm device is required.")

    smoke = _load_gemma4_smoke_module()
    model_path = args.model or smoke.resolve_default_model_path()
    if not model_path or not os.path.isdir(model_path):
        raise SystemExit("Gemma4 model path not found. Pass --model explicitly.")

    os.environ.setdefault("FASTINFERENCE_GEMMA4_LAYER_PROFILE", "1")
    os.environ.setdefault("FASTINFERENCE_KV_TYPE", args.kv_type)

    from vllm.model_executor.layers.quantization.tensor import (
        get_awq_runtime_stats,
        get_awq_runtime_prefix_stats,
        reset_awq_runtime_stats,
    )

    build_args = smoke._build_parser().parse_args(
        [
            "--model",
            model_path,
            "--max-new-tokens",
            str(max(args.max_new_tokens, args.warmup_decode + args.decode_steps + 8)),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--max-num-batched-tokens",
            str(args.max_num_batched_tokens),
            "--max-model-len",
            str(args.max_model_len),
        ]
    )
    if args.tokenizer is not None:
        build_args.tokenizer = args.tokenizer

    print(
        f"[Gemma4Profile] model={model_path} kv={os.environ['FASTINFERENCE_KV_TYPE']} "
        f"prompt_tokens≈{args.prompt_tokens} decode_steps={args.decode_steps}"
    )
    engine, tokenizer, load_s = smoke._build_engine(build_args)
    prompt = _build_prompt(tokenizer, min(args.prompt_tokens, args.max_model_len - 64))
    prompt_len = len(tokenizer.encode(prompt))

    reset_awq_runtime_stats()
    sp = SamplingParams(
        max_tokens=max(args.max_new_tokens, args.warmup_decode + args.decode_steps + 8),
        min_tokens=1,
        temperature=0.0,
        top_p=1.0,
    )
    engine.add_request("gemma4_profile", smoke._apply_chat_template(tokenizer, prompt), sp)

    prefill_ms: list[float] = []
    while not _prefill_done(engine):
        prefill_ms.append(_step_with_timing(engine))

    for _ in range(args.warmup_decode):
        if not _has_running_requests(engine):
            break
        _step_with_timing(engine)

    decode_ms: list[float] = []
    for _ in range(args.decode_steps):
        if not _has_running_requests(engine):
            break
        decode_ms.append(_step_with_timing(engine))

    while _has_running_requests(engine):
        engine.step()

    awq_stats = get_awq_runtime_stats()
    awq_prefix_stats = get_awq_runtime_prefix_stats(limit=12)
    summary = {
        "model": model_path,
        "load_s": round(load_s, 3),
        "prompt_tokens": prompt_len,
        "prefill_steps": len(prefill_ms),
        "prefill_step_mean_ms": round(_mean(prefill_ms), 3),
        "prefill_step_p50_ms": round(_p50(prefill_ms), 3),
        "decode_profiled_steps": len(decode_ms),
        "decode_step_mean_ms": round(_mean(decode_ms), 3),
        "decode_step_p50_ms": round(_p50(decode_ms), 3),
        "awq_stats": awq_stats,
        "awq_prefix_stats": awq_prefix_stats,
    }
    print("[Gemma4ProfileSummary] " + json.dumps(summary, ensure_ascii=True, sort_keys=True))
    print(
        "[Gemma4ProfileNote] Internal layer spans are emitted as "
        "`[Gemma4LayerProfile]` when the process exits."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
