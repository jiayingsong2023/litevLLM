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
import copy
import json
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from vllm.sampling_params import SamplingParams

_ROOT = Path(__file__).resolve().parents[2]


def _load_gemma4_smoke_module() -> Any:
    import importlib.util
    import sys

    module_path = _ROOT / "tests" / "tools" / "gemma4_single_prompt_smoke.py"
    spec = importlib.util.spec_from_file_location(
        "gemma4_single_prompt_smoke", module_path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_prompt(tokenizer: Any, target_tokens: int) -> str:
    sentence = (
        "Explain how to improve inference throughput for a memory-bound "
        "transformer model without sacrificing output quality. "
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


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _record_timer(
    stats: dict[str, dict[str, float]], name: str, elapsed_s: float
) -> None:
    row = stats.setdefault(name, {"time_s": 0.0, "count": 0.0})
    row["time_s"] += float(elapsed_s)
    row["count"] += 1.0


def _wrap_timed(
    obj: Any,
    method_name: str,
    stats: dict[str, dict[str, float]],
    timer_name: str,
) -> None:
    original = getattr(obj, method_name, None)
    if original is None:
        return

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        _sync_cuda()
        t0 = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            _sync_cuda()
            _record_timer(stats, timer_name, time.perf_counter() - t0)

    setattr(obj, method_name, wrapped)


def _install_runtime_timers(engine: Any) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    _wrap_timed(engine.step_scheduler, "build_plan", stats, "scheduler.build_plan")
    _wrap_timed(engine.execution_backend, "run_prefills", stats, "backend.run_prefills")
    _wrap_timed(engine.execution_backend, "run_decodes", stats, "backend.run_decodes")
    _wrap_timed(
        engine.execution_backend, "decode_step_sync", stats, "backend.decode_step_sync"
    )
    _wrap_timed(engine.prefill_executor, "execute", stats, "prefill_executor.execute")
    _wrap_timed(
        engine.decode_executor,
        "execute_sync_fast",
        stats,
        "decode_executor.execute_sync_fast",
    )
    _wrap_timed(
        engine.decode_executor, "execute_batch", stats, "decode_executor.execute_batch"
    )
    return stats


def _install_sampling_timer(engine: Any, stats: dict[str, dict[str, float]]) -> None:
    if getattr(engine, "sampling_driver", None) is not None:
        _wrap_timed(
            engine.sampling_driver,
            "sample_next_token",
            stats,
            "sampling.sample_next_token",
        )


def _stats_snapshot(rows: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        str(name): {
            "time_s": float(row.get("time_s", 0.0)),
            "count": float(row.get("count", 0.0)),
        }
        for name, row in rows.items()
    }


def _stats_diff(
    before: dict[str, dict[str, float]],
    after: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    names = sorted(set(before) | set(after))
    out: dict[str, dict[str, float]] = {}
    for name in names:
        time_s = float(after.get(name, {}).get("time_s", 0.0)) - float(
            before.get(name, {}).get("time_s", 0.0)
        )
        count = float(after.get(name, {}).get("count", 0.0)) - float(
            before.get(name, {}).get("count", 0.0)
        )
        if abs(time_s) > 1e-12 or abs(count) > 1e-12:
            out[name] = {"time_s": time_s, "count": count}
    return out


def _format_profile_rows(
    stats: dict[str, dict[str, float]],
    *,
    total_ms: float | None = None,
) -> dict[str, dict[str, int | float]]:
    rows: dict[str, dict[str, int | float]] = {}
    denom_ms = float(total_ms or 0.0)
    if denom_ms <= 0.0:
        denom_ms = sum(float(row.get("time_s", 0.0)) * 1000.0 for row in stats.values())
    for name, row in sorted(
        stats.items(),
        key=lambda item: float(item[1].get("time_s", 0.0)),
        reverse=True,
    ):
        count = int(row.get("count", 0.0))
        time_ms = float(row.get("time_s", 0.0)) * 1000.0
        rows[name] = {
            "time_ms": round(time_ms, 3),
            "count": count,
            "avg_ms": round(time_ms / count, 3) if count > 0 else 0.0,
            "share_of_step_wall_pct": round(time_ms / denom_ms * 100.0, 2)
            if denom_ms > 0
            else 0.0,
        }
    return rows


def _collect_layer_profile_snapshot() -> dict[str, dict[str, float]]:
    from vllm.model_executor.models import gemma4 as gemma4_model

    return _stats_snapshot(getattr(gemma4_model, "_GEMMA4_PROFILE_STATS", {}))


def _reset_layer_profile() -> None:
    from vllm.model_executor.models import gemma4 as gemma4_model

    getattr(gemma4_model, "_GEMMA4_PROFILE_STATS", {}).clear()


def _force_enable_layer_profile() -> None:
    from vllm.model_executor.models import gemma4 as gemma4_model

    gemma4_model.set_gemma4_tuning_config(
        {
            key: value
            for key, value in os.environ.items()
            if key.startswith("FASTINFERENCE_")
        },
        locked=True,
    )


def _collect_moe_kernel_snapshot() -> dict[str, dict[str, float]]:
    try:
        from vllm.kernels.triton.gemma4_moe_int4 import (
            get_moe_kernel_profile_stats,
        )

        return _stats_snapshot(get_moe_kernel_profile_stats())
    except Exception:
        return {}


def _reset_moe_kernel_profile() -> None:
    try:
        from vllm.kernels.triton.gemma4_moe_int4 import (
            reset_moe_kernel_profile_stats,
        )

        reset_moe_kernel_profile_stats()
    except Exception:
        return


def _profile_phase_breakdown(
    *,
    layer_before: dict[str, dict[str, float]],
    layer_after: dict[str, dict[str, float]],
    kernel_before: dict[str, dict[str, float]],
    kernel_after: dict[str, dict[str, float]],
    runtime_before: dict[str, dict[str, float]],
    runtime_after: dict[str, dict[str, float]],
    wall_ms: float,
) -> dict[str, Any]:
    layer = _stats_diff(layer_before, layer_after)
    kernel = _stats_diff(kernel_before, kernel_after)
    runtime = _stats_diff(runtime_before, runtime_after)
    return {
        "wall_ms_total": round(float(wall_ms), 3),
        "layer_spans": _format_profile_rows(layer, total_ms=wall_ms),
        "moe_kernel_spans": _format_profile_rows(kernel, total_ms=wall_ms),
        "runtime_spans": _format_profile_rows(runtime, total_ms=wall_ms),
    }


def _summarize_torch_profiler(prof: Any) -> dict[str, float | int]:
    launch_markers = (
        "cudaLaunchKernel",
        "cudaLaunchKernelEx",
        "hipLaunchKernel",
        "hipExtModuleLaunchKernel",
    )
    cpu_self_us = 0.0
    device_us = 0.0
    launch_cpu_us = 0.0
    launch_count = 0
    kernel_events = 0
    for event in prof.key_averages():
        key = str(getattr(event, "key", ""))
        cpu_self_us += float(getattr(event, "self_cpu_time_total", 0.0) or 0.0)
        event_device_us = float(
            getattr(
                event,
                "device_time_total",
                getattr(event, "cuda_time_total", 0.0),
            )
            or 0.0
        )
        if event_device_us > 0.0:
            device_us += event_device_us
            kernel_events += int(getattr(event, "count", 0) or 0)
        if any(marker in key for marker in launch_markers):
            launch_cpu_us += float(getattr(event, "self_cpu_time_total", 0.0) or 0.0)
            launch_count += int(getattr(event, "count", 0) or 0)
    return {
        "cpu_self_ms": round(cpu_self_us / 1000.0, 3),
        "device_kernel_ms": round(device_us / 1000.0, 3),
        "kernel_launch_cpu_ms": round(launch_cpu_us / 1000.0, 3),
        "kernel_launch_count": launch_count,
        "device_event_count": kernel_events,
    }


def _has_running_requests(engine: Any) -> bool:
    return int(getattr(engine, "active_request_count", 0)) > 0


def _projection_kind(prefix: str) -> str:
    suffixes = (
        ("self_attn.q_proj", "attn_q_proj"),
        ("self_attn.k_proj", "attn_k_proj"),
        ("self_attn.v_proj", "attn_v_proj"),
        ("self_attn.o_proj", "attn_o_proj"),
        ("mlp.gate_proj", "mlp_gate_proj"),
        ("mlp.up_proj", "mlp_up_proj"),
        ("mlp.down_proj", "mlp_down_proj"),
    )
    for marker, kind in suffixes:
        if marker in prefix:
            return kind
    return "other"


def _summarize_awq_projection_prefixes(
    prefix_stats: dict[str, dict[str, int]],
) -> dict[str, dict[str, int | float]]:
    buckets: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))  # type: ignore[arg-type]
    for prefix, stats in prefix_stats.items():
        kind = _projection_kind(prefix)
        bucket = buckets[kind]
        bucket["prefixes"] += 1
        for key, value in stats.items():
            bucket[key] += int(value)

    out: dict[str, dict[str, int | float]] = {}
    for kind, stats in sorted(buckets.items()):
        row: dict[str, int | float] = {key: int(value) for key, value in stats.items()}
        calls = int(stats.get("matmul_calls", 0))
        fused_success = int(stats.get("fused_success", 0))
        if calls:
            row["fused_success_rate"] = round(fused_success / calls, 4)
        out[kind] = row
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma4 LiteEngine layer profiler")
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model path (default: auto-discover local Gemma4 31B)",
    )
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup-decode", type=int, default=4)
    parser.add_argument("--decode-steps", type=int, default=24)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--kv-type", type=str, default="turbo_int4")
    parser.add_argument(
        "--torch-profiler",
        action="store_true",
        help="Collect PyTorch CPU/CUDA profiler summaries for prefill and decode.",
    )
    parser.add_argument(
        "--awq-prefix-limit",
        type=int,
        default=256,
        help="Number of per-layer AWQ prefixes to include in the JSON summary.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional path to write the machine-readable summary JSON.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm device is required.")

    smoke = _load_gemma4_smoke_module()
    model_path = args.model or smoke.resolve_default_model_path()
    if not model_path or not os.path.isdir(model_path):
        raise SystemExit("Gemma4 model path not found. Pass --model explicitly.")

    os.environ["FASTINFERENCE_GEMMA4_LAYER_PROFILE"] = "1"
    os.environ["FASTINFERENCE_GEMMA4_MOE_KERNEL_PROFILE"] = "1"
    os.environ.setdefault("FASTINFERENCE_KV_TYPE", args.kv_type)

    from vllm.model_executor.layers.quantization.tensor import (
        get_awq_runtime_prefix_stats,
        get_awq_runtime_stats,
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
    runtime_timer_stats = _install_runtime_timers(engine)
    _force_enable_layer_profile()
    _reset_layer_profile()
    _reset_moe_kernel_profile()
    prompt = _build_prompt(tokenizer, min(args.prompt_tokens, args.max_model_len - 64))
    prompt_len = len(tokenizer.encode(prompt))

    reset_awq_runtime_stats()
    sp = SamplingParams(
        max_tokens=max(args.max_new_tokens, args.warmup_decode + args.decode_steps + 8),
        min_tokens=1,
        temperature=0.0,
        top_p=1.0,
    )
    engine.add_request(
        "gemma4_profile", smoke._apply_chat_template(tokenizer, prompt), sp
    )
    _install_sampling_timer(engine, runtime_timer_stats)

    prefill_layer_before = _collect_layer_profile_snapshot()
    prefill_kernel_before = _collect_moe_kernel_snapshot()
    prefill_runtime_before = copy.deepcopy(runtime_timer_stats)
    prefill_profiler_summary: dict[str, float | int] = {}
    prefill_ms: list[float] = []
    prefill_profile_ctx = (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            with_stack=False,
        )
        if args.torch_profiler
        else nullcontext()
    )
    with prefill_profile_ctx as prof:
        while not _prefill_done(engine):
            prefill_ms.append(_step_with_timing(engine))
    if args.torch_profiler:
        prefill_profiler_summary = _summarize_torch_profiler(prof)
    prefill_layer_after = _collect_layer_profile_snapshot()
    prefill_kernel_after = _collect_moe_kernel_snapshot()
    prefill_runtime_after = copy.deepcopy(runtime_timer_stats)

    for _ in range(args.warmup_decode):
        if not _has_running_requests(engine):
            break
        _step_with_timing(engine)

    decode_layer_before = _collect_layer_profile_snapshot()
    decode_kernel_before = _collect_moe_kernel_snapshot()
    decode_runtime_before = copy.deepcopy(runtime_timer_stats)
    decode_profiler_summary: dict[str, float | int] = {}
    decode_ms: list[float] = []
    decode_profile_ctx = (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            with_stack=False,
        )
        if args.torch_profiler
        else nullcontext()
    )
    with decode_profile_ctx as prof:
        for _ in range(args.decode_steps):
            if not _has_running_requests(engine):
                break
            decode_ms.append(_step_with_timing(engine))
    if args.torch_profiler:
        decode_profiler_summary = _summarize_torch_profiler(prof)
    decode_layer_after = _collect_layer_profile_snapshot()
    decode_kernel_after = _collect_moe_kernel_snapshot()
    decode_runtime_after = copy.deepcopy(runtime_timer_stats)

    while _has_running_requests(engine):
        engine.step()

    awq_stats = get_awq_runtime_stats()
    awq_prefix_stats = get_awq_runtime_prefix_stats(limit=args.awq_prefix_limit)
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
        "phase_breakdown": {
            "prefill": _profile_phase_breakdown(
                layer_before=prefill_layer_before,
                layer_after=prefill_layer_after,
                kernel_before=prefill_kernel_before,
                kernel_after=prefill_kernel_after,
                runtime_before=prefill_runtime_before,
                runtime_after=prefill_runtime_after,
                wall_ms=sum(prefill_ms),
            ),
            "decode": _profile_phase_breakdown(
                layer_before=decode_layer_before,
                layer_after=decode_layer_after,
                kernel_before=decode_kernel_before,
                kernel_after=decode_kernel_after,
                runtime_before=decode_runtime_before,
                runtime_after=decode_runtime_after,
                wall_ms=sum(decode_ms),
            ),
        },
        "torch_profiler": {
            "enabled": bool(args.torch_profiler),
            "prefill": prefill_profiler_summary,
            "decode": decode_profiler_summary,
        },
        "awq_stats": awq_stats,
        "awq_projection_summary": _summarize_awq_projection_prefixes(awq_prefix_stats),
        "awq_prefix_stats": awq_prefix_stats,
    }
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        "[Gemma4ProfileSummary] "
        + json.dumps(summary, ensure_ascii=True, sort_keys=True)
    )
    print(
        "[Gemma4ProfileNote] Internal layer spans are emitted as "
        "`[Gemma4LayerProfile]` when the process exits."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
