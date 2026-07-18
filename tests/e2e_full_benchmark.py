# SPDX-License-Identifier: Apache-2.0
import argparse
import asyncio
import contextlib
import copy
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median
from typing import Any

import torch
from PIL import Image

from vllm import SamplingParams
from vllm.config import (
    CacheConfig,
    LoadConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.engine.async_llm import AsyncLLM


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_path: str
    display_name: str
    quant: str
    concurrent_reqs: int
    prompt_tokens_target: int
    max_new_tokens: int
    gpu_memory_utilization: float
    max_model_len: int
    max_run_seconds: int
    stable_env: dict[str, str]


@dataclass(frozen=True)
class BenchmarkRequestSpec:
    prompt: str
    multi_modal_data: dict[str, Any] | None = None


@dataclass(frozen=True)
class WarmupConfig:
    prefill_rounds: int = 1
    decode_rounds: int = 1
    decode_tokens: int = 8
    burst_rounds: int = 0
    burst_concurrency: int = 0
    burst_decode_tokens: int = 8


_GEMMA31B_SCHEDULER_PROFILES: dict[str, dict[str, str]] = {
    "baseline": {
        "FASTINFERENCE_LITE_PREFILL_CHUNK": "256",
        "FASTINFERENCE_LITE_PREFILL_MICROBATCH": "2",
        "FASTINFERENCE_LITE_PREFILL_RESERVED_TOKENS": "0",
        "FASTINFERENCE_LITE_PREFILL_RESERVE_BACKLOG": "2",
        "FASTINFERENCE_LITE_PREFILL_CATCHUP_RATIO": "0.25",
        "FASTINFERENCE_LITE_DECODE_PRIORITY": "1",
    },
    "decode_bias": {
        "FASTINFERENCE_LITE_PREFILL_CHUNK": "192",
        "FASTINFERENCE_LITE_PREFILL_MICROBATCH": "1",
        "FASTINFERENCE_LITE_PREFILL_RESERVED_TOKENS": "0",
        "FASTINFERENCE_LITE_PREFILL_RESERVE_BACKLOG": "3",
        "FASTINFERENCE_LITE_PREFILL_CATCHUP_RATIO": "0.20",
        "FASTINFERENCE_LITE_DECODE_PRIORITY": "1",
    },
    "catchup_prefill": {
        "FASTINFERENCE_LITE_PREFILL_CHUNK": "384",
        "FASTINFERENCE_LITE_PREFILL_MICROBATCH": "2",
        "FASTINFERENCE_LITE_PREFILL_RESERVED_TOKENS": "128",
        "FASTINFERENCE_LITE_PREFILL_RESERVE_BACKLOG": "2",
        "FASTINFERENCE_LITE_PREFILL_CATCHUP_RATIO": "0.35",
        "FASTINFERENCE_LITE_DECODE_PRIORITY": "1",
    },
}

_GEMMA4_12B_BATCH_ENV: dict[str, str] = {
    "FASTINFERENCE_KV_TYPE": "fp8",
    "FASTINFERENCE_FUSION_LEVEL": "2",
    "FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS": "4",
    "FASTINFERENCE_KV_MAX_MODEL_LEN": "512",
}

_DEEPSEEK_V4_FLASH_GGUF_PATH = (
    "models/DeepSeek-V4-Flash-ds4/"
    "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
)


# KV cache default: TurboQuant INT4 (FASTINFERENCE_KV_TYPE=turbo_int4).
MODEL_SPECS: dict[str, ModelSpec] = {
    "tinyllama": ModelSpec(
        key="tinyllama",
        model_path="models/TinyLlama-1.1B-Chat-v1.0",
        display_name="TinyLlama-1.1B (Dense)",
        quant="none",
        # BS=8, ~4096-token prefill per sequence (reduce concurrent if OOM on smaller GPUs).
        concurrent_reqs=8,
        prompt_tokens_target=4096,
        max_new_tokens=32,
        gpu_memory_utilization=0.92,
        max_model_len=4096,
        max_run_seconds=600,
        stable_env={"FASTINFERENCE_KV_TYPE": "turbo_int4"},
    ),
    "qwen35_9b_awq": ModelSpec(
        key="qwen35_9b_awq",
        model_path="models/Qwen3.5-9B-AWQ",
        display_name="Qwen3.5-9B (AWQ INT4)",
        quant="awq",
        # BS=8, ~4096-token prefill; FP8 KV; 48GB+ typical (use --qwen9b-concurrent to tune).
        concurrent_reqs=8,
        prompt_tokens_target=4096,
        max_new_tokens=24,
        gpu_memory_utilization=0.92,
        max_model_len=4096,
        max_run_seconds=960,
        stable_env={
            "FASTINFERENCE_KV_TYPE": "turbo_int4",
            "FASTINFERENCE_FUSION_LEVEL": "2",
            # Prefill: SDPA on first full-attn chunk (faster than eager HF matmul; set "0" for strict HF parity).
            "FASTINFERENCE_QWEN35_FULLATTN_USE_SDP_PREFILL": "1",
        },
    ),
    "gemma4_12b_awq": ModelSpec(
        key="gemma4_12b_awq",
        model_path=os.environ.get(
            "MODEL_GEMMA4_12B_AWQ", "models/gemma-4-12B-it-AWQ-INT4"
        ),
        display_name="Gemma4-12B (AWQ INT4, M=4)",
        quant="awq",
        concurrent_reqs=4,
        prompt_tokens_target=128,
        max_new_tokens=32,
        gpu_memory_utilization=0.80,
        max_model_len=512,
        max_run_seconds=1200,
        stable_env=dict(_GEMMA4_12B_BATCH_ENV),
    ),
    "gemma4_31b_q4": ModelSpec(
        key="gemma4_31b_q4",
        model_path=os.environ.get(
            "MODEL_GEMMA4_31B_Q4", "models/gemma-4-31B-it-AWQ-4bit"
        ),
        display_name="Gemma4-31B (Q4)",
        quant="compressed-tensors",
        concurrent_reqs=1,
        prompt_tokens_target=384,
        max_new_tokens=24,
        gpu_memory_utilization=0.90,
        max_model_len=512,
        max_run_seconds=1200,
        stable_env={},
    ),
    "gemma4_26b_a4b": ModelSpec(
        key="gemma4_26b_a4b",
        model_path=os.environ.get(
            "MODEL_GEMMA4_26B_A4B", "models/gemma-4-26B-A4B-it-AWQ-4bit"
        ),
        display_name="Gemma4-26B A4B (Q4 MoE)",
        quant="compressed-tensors",
        concurrent_reqs=1,
        prompt_tokens_target=384,
        max_new_tokens=24,
        gpu_memory_utilization=0.90,
        max_model_len=512,
        max_run_seconds=1200,
        stable_env={},
    ),
    "deepseek_v4_flash_q2_gguf": ModelSpec(
        key="deepseek_v4_flash_q2_gguf",
        model_path=os.environ.get(
            "MODEL_DEEPSEEK_V4_FLASH_GGUF",
            _DEEPSEEK_V4_FLASH_GGUF_PATH,
        ),
        display_name="DeepSeek V4 Flash Q2 GGUF",
        quant="deepseek-v4-flash-gguf",
        concurrent_reqs=1,
        # DeepSeek GGUF prefill is currently sequential token replay.
        # Keep the default smoke short; use --deepseek-prompt-tokens for long runs.
        prompt_tokens_target=32,
        max_new_tokens=16,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        max_run_seconds=1200,
        stable_env={},
    ),
}

# ROCm/HIP: large BS × long prefill + AWQ Triton + FP8 KV often triggers hipErrorLaunchFailure
# (often async OOM or kernel fault). Use conservative defaults unless aggressive mode is on.
_ROCM_E2E_SAFE_CONCURRENT = 4
_ROCM_E2E_SAFE_PROMPT_TOKENS = 1024
_ROCM_E2E_SAFE_GPU_MEM_UTIL = 0.85


def _is_rocm() -> bool:
    return bool(getattr(torch.version, "hip", None))


def _is_hf_repo_id(model_ref: str) -> bool:
    return "/" in model_ref and not model_ref.startswith("/")


def _is_e2e_aggressive(args: argparse.Namespace) -> bool:
    if getattr(args, "aggressive", False):
        return True
    raw = os.environ.get("FASTINFERENCE_E2E_AGGRESSIVE", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _clamp_non_negative_int(value: int, *, minimum: int = 0) -> int:
    return max(minimum, int(value))


def _read_int_with_default(raw: str | None, default: int) -> int:
    if raw is None or raw.strip() == "":
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _resolve_warmup_config(args: argparse.Namespace) -> WarmupConfig:
    warmup_preset = (
        str(
            os.environ.get(
                "FASTINFERENCE_BENCH_WARMUP_PRESET",
                getattr(args, "warmup_preset", "default"),
            )
        )
        .strip()
        .lower()
    )
    if warmup_preset not in ("default", "off", "cold"):
        warmup_preset = "default"
    if warmup_preset == "off":
        preset_prefill = 0
        preset_decode = 0
        preset_decode_tokens = 8
        preset_burst_rounds = 0
        preset_burst_concurrency = 0
        preset_burst_decode_tokens = 8
    elif warmup_preset == "cold":
        preset_prefill = 2
        preset_decode = 2
        preset_decode_tokens = 16
        preset_burst_rounds = 1
        preset_burst_concurrency = 2
        preset_burst_decode_tokens = 8
    else:
        preset_prefill = args.warmup_prefill_rounds
        preset_decode = args.warmup_decode_rounds
        preset_decode_tokens = args.warmup_decode_tokens
        preset_burst_rounds = args.warmup_burst_rounds
        preset_burst_concurrency = args.warmup_burst_concurrency
        preset_burst_decode_tokens = args.warmup_burst_decode_tokens

    prefill_rounds = _clamp_non_negative_int(
        _read_int_with_default(
            os.environ.get("FASTINFERENCE_BENCH_WARMUP_PREFILL_ROUNDS"),
            preset_prefill,
        )
    )
    decode_rounds = _clamp_non_negative_int(
        _read_int_with_default(
            os.environ.get("FASTINFERENCE_BENCH_WARMUP_DECODE_ROUNDS"),
            preset_decode,
        )
    )
    decode_tokens = _clamp_non_negative_int(
        _read_int_with_default(
            os.environ.get("FASTINFERENCE_BENCH_WARMUP_DECODE_TOKENS"),
            preset_decode_tokens,
        ),
        minimum=1,
    )
    burst_rounds = _clamp_non_negative_int(
        _read_int_with_default(
            os.environ.get("FASTINFERENCE_BENCH_WARMUP_BURST_ROUNDS"),
            preset_burst_rounds,
        )
    )
    burst_concurrency = _clamp_non_negative_int(
        _read_int_with_default(
            os.environ.get("FASTINFERENCE_BENCH_WARMUP_BURST_CONCURRENCY"),
            preset_burst_concurrency,
        )
    )
    burst_decode_tokens = _clamp_non_negative_int(
        _read_int_with_default(
            os.environ.get("FASTINFERENCE_BENCH_WARMUP_BURST_DECODE_TOKENS"),
            preset_burst_decode_tokens,
        ),
        minimum=1,
    )
    return WarmupConfig(
        prefill_rounds=prefill_rounds,
        decode_rounds=decode_rounds,
        decode_tokens=decode_tokens,
        burst_rounds=burst_rounds,
        burst_concurrency=burst_concurrency,
        burst_decode_tokens=burst_decode_tokens,
    )


def _resolve_compile_cache_env(
    args: argparse.Namespace,
) -> tuple[dict[str, str], dict[str, Any]]:
    raw = str(
        os.environ.get(
            "FASTINFERENCE_BENCH_COMPILE_CACHE_DIR",
            getattr(args, "compile_cache_dir", ""),
        )
    ).strip()
    if raw == "":
        return {}, {"enabled": False}
    cache_dir = Path(raw).expanduser().resolve()
    if bool(getattr(args, "compile_cache_clear", False)) and cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    inductor_dir = (cache_dir / "torchinductor").resolve()
    inductor_dir.mkdir(parents=True, exist_ok=True)
    env_map = {
        "TRITON_CACHE_DIR": str(cache_dir),
        "TORCHINDUCTOR_CACHE_DIR": str(inductor_dir),
    }
    return env_map, {
        "enabled": True,
        "cache_dir": str(cache_dir),
        "torchinductor_cache_dir": str(inductor_dir),
    }


def _cache_dir_stats(path: str | None) -> dict[str, Any]:
    if not path:
        return {"exists": False, "files": 0, "bytes": 0}
    p = Path(path)
    if not p.exists():
        return {"exists": False, "files": 0, "bytes": 0}
    files = 0
    total_bytes = 0
    for root, _dirs, names in os.walk(p):
        root_path = Path(root)
        for name in names:
            files += 1
            try:
                total_bytes += int((root_path / name).stat().st_size)
            except OSError:
                continue
    return {"exists": True, "files": files, "bytes": total_bytes}


def _format_targets(specs: list[ModelSpec]) -> str:
    return " + ".join(spec.display_name for spec in specs)


def _should_use_model_process_isolation(
    args: argparse.Namespace,
    model_keys: list[str],
) -> bool:
    if bool(getattr(args, "model_process_isolation", False)):
        return True
    if bool(getattr(args, "no_model_process_isolation", False)):
        return False
    if not _is_rocm():
        return False
    if len(model_keys) < 2:
        return False
    large_gemma = {
        "gemma4_31b_q4",
        "gemma4_26b_a4b",
        "deepseek_v4_flash_q2_gguf",
    }
    return any(key in large_gemma for key in model_keys)


_CLI_OPTS_WITH_VALUES = {
    "--models",
    "--json-out",
    "--runtime-stats-out",
    "--workload",
    "--multimodal-images-per-request",
    "--multimodal-image-size",
    "--tinyllama-concurrent",
    "--qwen9b-concurrent",
    "--gemma31b-concurrent",
    "--gemma31b-prompt-tokens",
    "--gemma31b-max-new-tokens",
    "--gemma31b-max-model-len",
    "--gemma26b-concurrent",
    "--gemma26b-prompt-tokens",
    "--gemma26b-max-new-tokens",
    "--gemma26b-max-model-len",
    "--warmup-prefill-rounds",
    "--warmup-preset",
    "--warmup-decode-rounds",
    "--warmup-decode-tokens",
    "--warmup-burst-rounds",
    "--warmup-burst-concurrency",
    "--warmup-burst-decode-tokens",
    "--compile-cache-dir",
    "--gemma31b-bucket-cutoff",
    "--gemma31b-bucket-decode-cutoff",
    "--gemma31b-bucket-short-profile",
    "--gemma31b-bucket-short-decode-profile",
    "--gemma31b-bucket-long-profile",
    "--gemma31b-schedule-profile",
    "--perf-baseline-json",
    "--perf-warn-min-tps-ratio",
    "--perf-warn-max-latency-ratio",
    "--perf-fail-on-regression",
}


def _build_child_cli_args(single_model_key: str, json_out: str) -> list[str]:
    child_args: list[str] = []
    argv = sys.argv[1:]
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token in (
            "--json-out",
            "--runtime-stats-out",
            "--models",
            "--model-process-isolation",
            "--no-model-process-isolation",
        ):
            if token in _CLI_OPTS_WITH_VALUES and idx + 1 < len(argv):
                idx += 2
            else:
                idx += 1
            continue
        if token.startswith("--json-out="):
            idx += 1
            continue
        if token.startswith("--runtime-stats-out="):
            idx += 1
            continue
        if token.startswith("--models="):
            idx += 1
            continue
        child_args.append(token)
        idx += 1
    child_args.extend(
        [
            "--models",
            single_model_key,
            "--json-out",
            json_out,
            "--no-model-process-isolation",
        ]
    )
    return child_args


def _run_isolated_model_benchmarks(
    model_keys: list[str],
    compile_cache_env: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, object]]]:
    summary: dict[str, dict[str, Any]] = {}
    runtime_stats_summary: dict[str, dict[str, object]] = {}
    base_env = os.environ.copy()
    base_env.update(compile_cache_env)
    script_path = Path(__file__).resolve()

    for key in model_keys:
        with tempfile.NamedTemporaryFile(
            prefix=f"fastinference_e2e_{key}_",
            suffix=".json",
            delete=False,
        ) as tmp:
            child_json = tmp.name
        child_args = _build_child_cli_args(key, child_json)
        print(f"[Isolated] Launching child benchmark for {key}")
        proc = subprocess.run(
            [sys.executable, str(script_path), *child_args],
            cwd=str(script_path.parents[1]),
            env=base_env,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Child benchmark failed for {key} with rc={proc.returncode}. "
                "Run the single-model command to inspect the detailed traceback."
            )
        with open(child_json, encoding="utf-8") as f:
            payload = json.load(f)
        with contextlib.suppress(OSError):
            os.unlink(child_json)
        child_summary = payload.get("summary", {})
        child_runtime = payload.get("runtime_stats", {})
        if key not in child_summary:
            raise RuntimeError(f"Child benchmark JSON for {key} missing summary entry.")
        summary[key] = dict(child_summary[key])
        runtime_stats_summary[key] = dict(child_runtime.get(key, {}))
    return summary, runtime_stats_summary


def _resolve_gemma31b_bucket_policy(
    spec: ModelSpec,
    args: argparse.Namespace,
) -> tuple[ModelSpec, dict[str, Any]]:
    enabled_raw = (
        str(
            os.environ.get(
                "FASTINFERENCE_GEMMA31B_BUCKET_ENABLE",
                "1" if bool(getattr(args, "gemma31b_bucket_enable", True)) else "0",
            )
        )
        .strip()
        .lower()
    )
    enabled = enabled_raw in ("1", "true", "yes", "on")
    short_profile = str(
        os.environ.get(
            "FASTINFERENCE_GEMMA31B_BUCKET_SHORT_PROFILE",
            getattr(args, "gemma31b_bucket_short_profile", "decode_bias"),
        )
    ).strip()
    short_decode_profile = str(
        os.environ.get(
            "FASTINFERENCE_GEMMA31B_BUCKET_SHORT_DECODE_PROFILE",
            getattr(
                args,
                "gemma31b_bucket_short_decode_profile",
                "catchup_prefill",
            ),
        )
    ).strip()
    long_profile = str(
        os.environ.get(
            "FASTINFERENCE_GEMMA31B_BUCKET_LONG_PROFILE",
            getattr(args, "gemma31b_bucket_long_profile", "baseline"),
        )
    ).strip()
    cutoff = int(
        _read_int_with_default(
            os.environ.get("FASTINFERENCE_GEMMA31B_BUCKET_CUTOFF"),
            int(getattr(args, "gemma31b_bucket_cutoff", 128)),
        )
    )
    decode_cutoff = int(
        _read_int_with_default(
            os.environ.get("FASTINFERENCE_GEMMA31B_BUCKET_DECODE_CUTOFF"),
            int(getattr(args, "gemma31b_bucket_decode_cutoff", 32)),
        )
    )
    profile_override = str(
        os.environ.get(
            "FASTINFERENCE_GEMMA31B_SCHED_PROFILE",
            getattr(args, "gemma31b_schedule_profile", "auto_bucket"),
        )
    ).strip()
    if short_profile not in _GEMMA31B_SCHEDULER_PROFILES:
        short_profile = "decode_bias"
    if short_decode_profile not in _GEMMA31B_SCHEDULER_PROFILES:
        short_decode_profile = "catchup_prefill"
    if long_profile not in _GEMMA31B_SCHEDULER_PROFILES:
        long_profile = "baseline"
    if profile_override not in (
        "auto_bucket",
        *tuple(_GEMMA31B_SCHEDULER_PROFILES.keys()),
    ):
        profile_override = "auto_bucket"
    if profile_override == "auto_bucket":
        if enabled:
            if int(spec.prompt_tokens_target) <= int(cutoff):
                selected_profile = (
                    short_decode_profile
                    if int(spec.max_new_tokens) <= int(decode_cutoff)
                    else short_profile
                )
            else:
                selected_profile = long_profile
        else:
            selected_profile = "baseline"
    else:
        selected_profile = profile_override
    merged_env = dict(spec.stable_env)
    merged_env.update(_GEMMA31B_SCHEDULER_PROFILES[selected_profile])
    merged_spec = replace(spec, stable_env=merged_env)
    policy = {
        "enabled": enabled,
        "mode": profile_override,
        "selected_profile": selected_profile,
        "cutoff_prompt_tokens": int(cutoff),
        "cutoff_decode_tokens": int(decode_cutoff),
        "short_profile": short_profile,
        "short_decode_profile": short_decode_profile,
        "long_profile": long_profile,
        "prompt_tokens_target": int(spec.prompt_tokens_target),
        "max_new_tokens": int(spec.max_new_tokens),
    }
    return merged_spec, policy


def _clear_prefix_cache_after_warmup(args: argparse.Namespace) -> bool:
    return not bool(getattr(args, "reuse_warmup_prefix_cache", False))


def _maybe_apply_rocm_safe_profile(spec: ModelSpec, aggressive: bool) -> ModelSpec:
    """Downgrade batch/prompt on AMD ROCm unless aggressive mode is enabled."""
    if aggressive or not _is_rocm():
        return spec
    if spec.key not in (
        "tinyllama",
        "qwen35_9b_awq",
        "gemma4_31b_q4",
        "gemma4_26b_a4b",
    ):
        return spec
    new_conc = min(spec.concurrent_reqs, _ROCM_E2E_SAFE_CONCURRENT)
    new_prompt = min(spec.prompt_tokens_target, _ROCM_E2E_SAFE_PROMPT_TOKENS)
    new_util = min(spec.gpu_memory_utilization, _ROCM_E2E_SAFE_GPU_MEM_UTIL)
    return replace(
        spec,
        concurrent_reqs=new_conc,
        prompt_tokens_target=new_prompt,
        gpu_memory_utilization=new_util,
        max_run_seconds=max(spec.max_run_seconds, 120 * new_conc),
    )


def _read_q4_group_size_and_bits(model_path: str) -> tuple[int, int]:
    config_path = os.path.join(model_path, "config.json")
    group_size, bits = 128, 4
    try:
        if not os.path.isfile(config_path):
            return group_size, bits
        with open(config_path, encoding="utf-8") as f:
            raw = json.load(f)
        qc = raw.get("quantization_config") or raw.get("compression_config") or {}
        groups = qc.get("config_groups")
        if isinstance(groups, dict):
            for g in groups.values():
                if not isinstance(g, dict):
                    continue
                w = g.get("weights")
                if isinstance(w, dict):
                    if w.get("group_size") is not None:
                        group_size = int(w["group_size"])
                    if w.get("num_bits") is not None:
                        bits = int(w["num_bits"])
                    break
        if qc.get("group_size") is not None:
            group_size = int(qc["group_size"])
        if qc.get("bits") is not None:
            bits = int(qc["bits"])
    except Exception as exc:
        print(f"  [Warn] Failed to parse Q4 config ({config_path}): {exc}")
    return group_size, bits


def _build_prompt(tokenizer, target_tokens: int) -> str:
    sentence = (
        "Please explain how modern AI systems improve software performance and reliability "
        "in practical engineering workflows. "
    )
    target_tokens = max(8, int(target_tokens))
    repeat = max(8, target_tokens // 12)
    prompt_text = sentence * repeat
    token_ids = tokenizer.encode(prompt_text)
    if len(token_ids) < target_tokens:
        # Extend prompt until the target token budget is reached.
        while len(token_ids) < target_tokens:
            prompt_text = prompt_text + sentence
            token_ids = tokenizer.encode(prompt_text)
    elif len(token_ids) > target_tokens:
        token_ids = token_ids[:target_tokens]
        prompt_text = tokenizer.decode(token_ids)
    return prompt_text


def _write_benchmark_image(
    image_path: Path, *, request_index: int, image_index: int, size: int
) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    color = (
        (64 + request_index * 23 + image_index * 11) % 256,
        (32 + request_index * 47 + image_index * 19) % 256,
        (128 + request_index * 13 + image_index * 29) % 256,
    )
    Image.new("RGB", (size, size), color=color).save(image_path)


def _build_benchmark_requests(
    *,
    prompt: str,
    request_count: int,
    workload: str,
    image_root: Path | None = None,
    multimodal_images_per_request: int = 1,
    multimodal_image_size: int = 8,
) -> list[BenchmarkRequestSpec]:
    requests: list[BenchmarkRequestSpec] = []
    for request_index in range(request_count):
        request_prompt = f"{prompt}\n[benchmark_request_id={request_index}]"
        multi_modal_data: dict[str, Any] | None = None
        if workload != "text":
            if image_root is None:
                raise ValueError("image_root is required for multimodal workloads")
            image_count = (
                1
                if workload == "multimodal_single"
                else max(2, int(multimodal_images_per_request))
            )
            image_blocks: list[dict[str, str]] = []
            for image_index in range(image_count):
                image_path = (
                    image_root / f"req_{request_index:03d}_img_{image_index:02d}.png"
                )
                _write_benchmark_image(
                    image_path,
                    request_index=request_index,
                    image_index=image_index,
                    size=max(2, int(multimodal_image_size)),
                )
                image_blocks.append({"image": f"file://{image_path}"})
            multi_modal_data = {"image": image_blocks}
        requests.append(
            BenchmarkRequestSpec(
                prompt=request_prompt,
                multi_modal_data=multi_modal_data,
            )
        )
    return requests


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(0.95 * (len(ordered) - 1))
    return ordered[idx]


def _finite_values(values: list[float]) -> list[float]:
    return [
        v for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))
    ]


def _fmt_float(value: float, fmt: str) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return format(float(value), fmt)
    return "n/a"


def _format_profile_summary(profile_stats: dict[str, object]) -> str:
    return (
        "PROFILE "
        f"requested={profile_stats.get('requested', 'unknown')} "
        f"effective={profile_stats.get('effective', 'unknown')} "
        f"kv={profile_stats.get('kv_cache_dtype', 'unknown')}"
    )


def _apply_temp_env(env_map: dict[str, str]) -> dict[str, str | None]:
    old_env: dict[str, str | None] = {}
    for key, value in env_map.items():
        old_env[key] = os.environ.get(key)
        os.environ[key] = value
    return old_env


def _restore_env(old_env: dict[str, str | None]) -> None:
    for key, value in old_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _collect_runtime_stats(
    llm: AsyncLLM,
    *,
    phase: str,
) -> dict[str, object]:
    snapshot = copy.deepcopy(llm.stats())
    profile_stats = dict(snapshot.get("profile") or {})
    derived_metrics = _derive_runtime_metrics(snapshot)
    return {
        "phase": phase,
        "collected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "profile": profile_stats,
        "stats": snapshot,
        "derived_metrics": derived_metrics,
    }


def _derive_runtime_metrics(snapshot: dict[str, object]) -> dict[str, object]:
    """Return only metrics emitted by the lite runtime."""
    observer = snapshot.get("observer", {})
    backend = snapshot.get("backend", {})
    async_driver = snapshot.get("async_driver", {})
    if not isinstance(observer, dict) or not isinstance(backend, dict):
        return {}

    observer_prefix = observer.get("prefix_cache", {})
    backend_prefix = backend.get("prefix_cache", {})
    if not isinstance(observer_prefix, dict):
        observer_prefix = {}
    if not isinstance(backend_prefix, dict):
        backend_prefix = {}
    if not isinstance(async_driver, dict):
        async_driver = {}

    request_count = float(observer_prefix.get("events", 0) or 0)
    step_count = float(observer.get("step_count", 0) or 0)
    async_steps = float(async_driver.get("steps", 0) or 0)
    async_backpressure_sleeps = float(async_driver.get("backpressure_sleeps", 0) or 0)
    async_idle_waits = float(async_driver.get("idle_waits", 0) or 0)
    async_background_errors = float(async_driver.get("background_errors", 0) or 0)
    materialized_hits = float(backend.get("prefix_cache_materialized_hits", 0) or 0)
    materialized_saved = float(
        backend.get("prefix_cache_materialized_saved_prefill_tokens", 0) or 0
    )
    lookup_comparisons = float(backend_prefix.get("lookup_comparisons", 0) or 0)
    lookup_candidates_total = float(
        backend_prefix.get("lookup_candidates_total", 0) or 0
    )
    return {
        "async_driver": {
            "steps": async_steps,
            "backpressure_sleeps": async_backpressure_sleeps,
            "idle_waits": async_idle_waits,
            "background_errors": async_background_errors,
            "min_step_interval_s": float(
                async_driver.get("min_step_interval_s", 0.0) or 0.0
            ),
            "backpressure_sleep_rate": (
                async_backpressure_sleeps / async_steps if async_steps else 0.0
            ),
            "idle_wait_rate": async_idle_waits / async_steps if async_steps else 0.0,
            "background_error_rate": (
                async_background_errors / async_steps if async_steps else 0.0
            ),
            "observer_step_gap": async_steps - step_count,
        },
        "prefix_cache": {
            "request_count": request_count,
            "lookup_hit_rate": float(backend_prefix.get("hit_rate", 0.0) or 0.0),
            "materialized_hit_rate": (
                materialized_hits / request_count if request_count else 0.0
            ),
            "saved_prefill_tokens_per_request": (
                materialized_saved / request_count if request_count else 0.0
            ),
            "saved_prefill_tokens_per_materialized_hit": (
                materialized_saved / materialized_hits if materialized_hits else 0.0
            ),
            "lookup_cost_per_request": (
                lookup_comparisons / request_count if request_count else 0.0
            ),
            "lookup_candidates_per_request": (
                lookup_candidates_total / request_count if request_count else 0.0
            ),
        },
    }


def _derive_runtime_phase_diffs(
    phases: dict[str, dict[str, object]],
) -> dict[str, object]:
    warmup = phases.get("warmup", {})
    benchmark = phases.get("benchmark", {})
    if not isinstance(warmup, dict) or not isinstance(benchmark, dict):
        return {}

    warmup_metrics = warmup.get("derived_metrics", {})
    benchmark_metrics = benchmark.get("derived_metrics", {})
    if not isinstance(warmup_metrics, dict) or not isinstance(benchmark_metrics, dict):
        return {}

    sections = ("async_driver", "prefix_cache")
    phase_diffs: dict[str, object] = {}
    for section in sections:
        warmup_section = warmup_metrics.get(section, {})
        benchmark_section = benchmark_metrics.get(section, {})
        if not isinstance(warmup_section, dict):
            warmup_section = {}
        if not isinstance(benchmark_section, dict):
            benchmark_section = {}
        keys = sorted(set(warmup_section) | set(benchmark_section))
        phase_diffs[section] = {
            "warmup": dict(warmup_section),
            "benchmark": dict(benchmark_section),
            "benchmark_delta": {
                key: float(benchmark_section.get(key, 0.0) or 0.0)
                - float(warmup_section.get(key, 0.0) or 0.0)
                for key in keys
                if isinstance(benchmark_section.get(key, 0.0), (int, float))
                and isinstance(warmup_section.get(key, 0.0), (int, float))
            },
        }
    return phase_diffs


def _format_runtime_phase_diff_summary(
    phase_diffs: dict[str, object],
) -> list[str]:
    lines: list[str] = []
    for section in ("async_driver", "prefix_cache"):
        diff = phase_diffs.get(section, {})
        if not isinstance(diff, dict):
            continue
        delta = diff.get("benchmark_delta", {})
        if not isinstance(delta, dict):
            continue
        rendered = ", ".join(
            f"{key}={float(value):+.3f}"
            for key, value in delta.items()
            if isinstance(value, (int, float))
        )
        if rendered:
            name = "async" if section == "async_driver" else section
            lines.append(f"  RUNTIME({name}): {rendered}")
    return lines


def _format_runtime_snapshot_summary(snapshot: dict[str, object]) -> list[str]:
    metrics = snapshot.get("derived_metrics", {})
    if not isinstance(metrics, dict):
        return []
    lines: list[str] = []
    for section in ("async_driver", "prefix_cache"):
        values = metrics.get(section, {})
        if not isinstance(values, dict):
            continue
        rendered = ", ".join(
            f"{key}={float(value):.3f}"
            for key, value in values.items()
            if isinstance(value, (int, float))
        )
        if rendered:
            lines.append(f"RUNTIME({section}): {rendered}")
    return lines


def _derive_awq_metrics(stats: dict[str, int]) -> dict[str, float]:
    attempts = float(stats.get("awq_fused_attempt", 0))
    success = float(stats.get("awq_fused_success", 0))
    ratio = (success / attempts) if attempts > 0 else 0.0
    return {
        "awq_fused_attempt": attempts,
        "awq_fused_success": success,
        "awq_fused_ratio": ratio,
        "awq_cache_hits": float(stats.get("awq_cache_hits", 0)),
        "awq_cache_misses": float(stats.get("awq_cache_misses", 0)),
        "awq_dense_cache_bytes_peak": float(stats.get("cache_bytes", 0)),
    }


def _effective_prompt_budget(spec: ModelSpec) -> int:
    """Upper bound on prompt tokens used in the benchmark (matches run_benchmark clamp)."""
    return min(
        int(spec.prompt_tokens_target),
        max(8, int(spec.max_model_len) - int(spec.max_new_tokens) - 1),
    )


def _build_vllm_config(spec: ModelSpec) -> VllmConfig:
    model_cfg = ModelConfig(
        model=spec.model_path,
        tokenizer=spec.model_path,
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="float16",
        max_model_len=spec.max_model_len,
    )
    # Allow concurrent long-prefill batches (e.g. BS=8 × CTX≈4096); do not cap at 8192.
    prompt_cap = _effective_prompt_budget(spec)
    max_num_batched_tokens = min(
        262144,
        max(8192, int(spec.concurrent_reqs) * prompt_cap),
    )
    scheduler_cfg = SchedulerConfig(
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=spec.concurrent_reqs,
        max_model_len=spec.max_model_len,
    )
    cache_cfg = CacheConfig(
        block_size=8,
        gpu_memory_utilization=spec.gpu_memory_utilization,
        swap_space=0,
    )
    v_cfg = VllmConfig(
        model_config=model_cfg,
        cache_config=cache_cfg,
        scheduler_config=scheduler_cfg,
        load_config=LoadConfig(load_format="auto"),
        quant_config=None,
    )
    if spec.quant == "awq":
        from vllm.model_executor.layers.quantization.awq import AWQConfig

        group_size, bits = _read_q4_group_size_and_bits(spec.model_path)
        v_cfg.quant_config = AWQConfig(weight_bits=bits, group_size=group_size)
    elif spec.quant == "compressed-tensors":
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsConfig,
        )

        group_size, bits = _read_q4_group_size_and_bits(spec.model_path)
        v_cfg.quant_config = CompressedTensorsConfig(
            weight_bits=bits, group_size=group_size
        )
    return v_cfg


async def _run_single_request(
    llm: AsyncLLM,
    request_id: str,
    prompt: str,
    prompt_tokens: int,
    sampling_params: SamplingParams,
    multi_modal_data: dict[str, Any] | None = None,
) -> dict[str, float]:
    start = time.perf_counter()
    token_timestamps: list[float] = []
    output_event_count = 0
    progressive_token_events = 0
    prev_token_count = 0
    generated_tokens = 0
    async for output in llm.generate(
        prompt,
        sampling_params,
        request_id,
        multi_modal_data=multi_modal_data,
    ):
        output_event_count += 1
        if not output.outputs:
            continue
        now = time.perf_counter()
        current_count = len(output.outputs[0].token_ids)
        generated_tokens = max(generated_tokens, current_count)
        new_tokens = max(0, current_count - prev_token_count)
        if new_tokens <= 0 and prev_token_count == 0 and current_count > 0:
            # Defensive fallback for engines that only emit a final output.
            new_tokens = current_count
        if new_tokens > 0:
            progressive_token_events += 1
            token_timestamps.extend([now] * new_tokens)
            prev_token_count = current_count
    end = time.perf_counter()
    first_token_at = token_timestamps[0] if token_timestamps else None
    last_token_at = token_timestamps[-1] if token_timestamps else None
    ttft_ms = (
        ((first_token_at - start) * 1000.0)
        if first_token_at is not None
        else float("nan")
    )
    e2e_ms = (end - start) * 1000.0
    decode_tokens = max(0, generated_tokens - 1)
    decode_ms = (
        (last_token_at - first_token_at) * 1000.0
        if first_token_at is not None
        and last_token_at is not None
        and len(token_timestamps) >= 2
        else float("nan")
    )
    # Sub-millisecond decode windows are often dominated by scheduling/buffering noise.
    if math.isfinite(decode_ms) and decode_ms < 1.0:
        decode_ms = float("nan")
    decode_tps = (
        (float(decode_tokens) * 1000.0 / decode_ms)
        if decode_tokens > 0 and math.isfinite(decode_ms) and decode_ms > 0.0
        else float("nan")
    )
    prefill_tps = (
        (float(prompt_tokens) * 1000.0 / ttft_ms)
        if prompt_tokens > 0 and math.isfinite(ttft_ms) and ttft_ms > 0.0
        else float("nan")
    )
    return {
        "tokens": float(generated_tokens),
        "prompt_tokens": float(prompt_tokens),
        "ttft_ms": ttft_ms,
        "e2e_ms": e2e_ms,
        "decode_tokens": float(decode_tokens),
        "decode_ms": decode_ms,
        "decode_tps": decode_tps,
        "prefill_tps": prefill_tps,
        "stream_output_events": float(output_event_count),
        "stream_progressive_token_events": float(progressive_token_events),
        "stream_token_events": float(len(token_timestamps)),
        "stream_progressive_visible": (
            1.0 if progressive_token_events > 1 and len(token_timestamps) >= 2 else 0.0
        ),
    }


async def _run_warmup_stage_request(
    llm: AsyncLLM,
    *,
    prompt: str,
    request_id: str,
    max_tokens: int,
    multi_modal_data: dict[str, Any] | None,
) -> float:
    start = time.perf_counter()
    async for _ in llm.generate(
        prompt,
        SamplingParams(max_tokens=max(1, int(max_tokens)), temperature=0.0),
        request_id,
        multi_modal_data=multi_modal_data,
    ):
        pass
    return (time.perf_counter() - start) * 1000.0


async def _run_benchmark_warmup(
    llm: AsyncLLM,
    *,
    spec: ModelSpec,
    request_specs: list[BenchmarkRequestSpec],
    warmup_config: WarmupConfig,
) -> list[dict[str, float | str]]:
    warmup_request = request_specs[0]
    warmup_trace: list[dict[str, float | str]] = []
    step = 0
    if warmup_config.prefill_rounds > 0:
        print(f"  [Warmup] Prefill compile rounds: {warmup_config.prefill_rounds}x")
    for _ in range(warmup_config.prefill_rounds):
        elapsed_ms = await _run_warmup_stage_request(
            llm,
            prompt=warmup_request.prompt,
            request_id=f"{spec.key}_warmup_prefill_{step}",
            max_tokens=1,
            multi_modal_data=warmup_request.multi_modal_data,
        )
        warmup_trace.append(
            {"stage": "prefill", "step": float(step), "elapsed_ms": elapsed_ms}
        )
        step += 1

    if warmup_config.decode_rounds > 0:
        print(
            "  [Warmup] Decode compile rounds: "
            f"{warmup_config.decode_rounds}x (max_tokens={warmup_config.decode_tokens})"
        )
    for _ in range(warmup_config.decode_rounds):
        elapsed_ms = await _run_warmup_stage_request(
            llm,
            prompt=warmup_request.prompt,
            request_id=f"{spec.key}_warmup_decode_{step}",
            max_tokens=warmup_config.decode_tokens,
            multi_modal_data=warmup_request.multi_modal_data,
        )
        warmup_trace.append(
            {"stage": "decode", "step": float(step), "elapsed_ms": elapsed_ms}
        )
        step += 1

    burst_concurrency = min(
        int(spec.concurrent_reqs),
        max(0, int(warmup_config.burst_concurrency)),
        len(request_specs),
    )
    if warmup_config.burst_rounds > 0 and burst_concurrency > 1:
        print(
            "  [Warmup] Burst jitter rounds: "
            f"{warmup_config.burst_rounds}x (concurrency={burst_concurrency}, "
            f"max_tokens={warmup_config.burst_decode_tokens})"
        )
    for round_idx in range(warmup_config.burst_rounds):
        start = time.perf_counter()
        tasks = [
            _run_warmup_stage_request(
                llm,
                prompt=request_specs[idx].prompt,
                request_id=f"{spec.key}_warmup_burst_{round_idx}_{idx}_{step}",
                max_tokens=warmup_config.burst_decode_tokens,
                multi_modal_data=request_specs[idx].multi_modal_data,
            )
            for idx in range(burst_concurrency)
        ]
        if tasks:
            await asyncio.gather(*tasks)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            warmup_trace.append(
                {"stage": "burst", "step": float(step), "elapsed_ms": elapsed_ms}
            )
            step += 1
    return warmup_trace


async def run_benchmark(
    spec: ModelSpec,
    *,
    workload: str = "text",
    multimodal_images_per_request: int = 2,
    multimodal_image_size: int = 8,
    warmup_config: WarmupConfig | None = None,
    fixed_decode_len: bool = True,
    clear_prefix_cache_after_warmup: bool = True,
) -> dict[str, Any]:
    print(f"\n{'=' * 72}")
    print(f"BENCHMARKING: {spec.display_name}")
    print(f"PATH: {spec.model_path}")
    print(
        "SETUP: "
        f"concurrency={spec.concurrent_reqs}, prompt_tokens~{spec.prompt_tokens_target}, "
        f"max_new_tokens={spec.max_new_tokens}, quant={spec.quant}, workload={workload}"
    )
    if spec.stable_env:
        env_summary = ", ".join(
            f"{key}={value}" for key, value in sorted(spec.stable_env.items())
        )
        print(f"STABLE_ENV: {env_summary}")
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(
            "GPU MEMORY: "
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated / {total_mem:.2f} GB total"
        )
    print(f"{'=' * 72}")

    if spec.quant == "deepseek-v4-flash-gguf":
        return _run_deepseek_v4_flash_gguf_benchmark(spec)

    if not os.path.isdir(spec.model_path) and not _is_hf_repo_id(spec.model_path):
        print("  [Skip] Model directory not found.")
        return {"skipped": 1.0}

    old_env = _apply_temp_env(spec.stable_env)
    llm: AsyncLLM | None = None
    runtime_stats_by_phase: dict[str, dict[str, object]] = {}
    warmup_trace: list[dict[str, float | str]] = []
    image_tmpdir: tempfile.TemporaryDirectory[str] | None = None
    try:
        if spec.quant in ("awq", "compressed-tensors"):
            from vllm.model_executor.layers.quantization.tensor import (
                get_awq_runtime_stats,
                reset_awq_runtime_stats,
            )

            reset_awq_runtime_stats()
        v_cfg = _build_vllm_config(spec)
        llm = AsyncLLM.from_vllm_config(v_cfg)
        if workload != "text" and not bool(
            getattr(llm.engine.multimodal_processor, "supports_multimodal", False)
        ):
            print("  [Skip] Model does not support multimodal benchmark workload.")
            return {
                "skipped": 1.0,
                "skip_reason": "multimodal_unsupported",
                "workload": {
                    "kind": workload,
                    "multimodal_images_per_request": (
                        1
                        if workload == "multimodal_single"
                        else max(2, int(multimodal_images_per_request))
                    ),
                    "multimodal_image_size": max(2, int(multimodal_image_size)),
                },
            }
        sampling_params = SamplingParams(
            max_tokens=spec.max_new_tokens,
            min_tokens=spec.max_new_tokens if fixed_decode_len else 0,
            ignore_eos=fixed_decode_len,
            temperature=0.0,
        )
        # Reserve decode room so prefill does not consume the full KV capacity.
        prompt_budget = min(
            int(spec.prompt_tokens_target),
            max(8, int(spec.max_model_len) - int(spec.max_new_tokens) - 1),
        )
        prompt = _build_prompt(llm.engine.tokenizer, prompt_budget)
        request_specs: list[BenchmarkRequestSpec]
        if workload == "text":
            request_specs = _build_benchmark_requests(
                prompt=prompt,
                request_count=spec.concurrent_reqs,
                workload=workload,
            )
        else:
            image_tmpdir = tempfile.TemporaryDirectory(prefix="fastinference_mm_bench_")
            request_specs = _build_benchmark_requests(
                prompt=prompt,
                request_count=spec.concurrent_reqs,
                workload=workload,
                image_root=Path(image_tmpdir.name),
                multimodal_images_per_request=multimodal_images_per_request,
                multimodal_image_size=multimodal_image_size,
            )
        prompt_tokens_by_request = [
            len(llm.engine.tokenizer.encode(req.prompt)) for req in request_specs
        ]

        resolved_warmup = warmup_config or WarmupConfig()
        warmup_trace = await _run_benchmark_warmup(
            llm,
            spec=spec,
            request_specs=request_specs,
            warmup_config=resolved_warmup,
        )
        runtime_stats_by_phase["warmup"] = _collect_runtime_stats(llm, phase="warmup")
        llm.reset_stats(clear_prefix_cache=clear_prefix_cache_after_warmup)

        print("  [Run] Launching concurrent benchmark requests...")
        wall_start = time.perf_counter()
        tasks = [
            _run_single_request(
                llm=llm,
                request_id=f"{spec.key}_{idx}_{int(time.time())}",
                prompt=request_specs[idx].prompt,
                prompt_tokens=prompt_tokens_by_request[idx],
                sampling_params=sampling_params,
                multi_modal_data=request_specs[idx].multi_modal_data,
            )
            for idx in range(spec.concurrent_reqs)
        ]
        try:
            request_results = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=float(spec.max_run_seconds),
            )
        except TimeoutError:
            awq_stats: dict = {}
            if spec.quant in ("awq", "compressed-tensors"):
                from vllm.model_executor.layers.quantization.tensor import (
                    get_awq_runtime_stats,
                )

                awq_stats = get_awq_runtime_stats()
            print(
                f"  [Timeout] Exceeded {spec.max_run_seconds}s for {spec.display_name}; "
                "marking as timeout in summary."
            )
            if awq_stats:
                print(
                    f"  AWQ_STATS(timeout): {json.dumps(awq_stats, ensure_ascii=True, sort_keys=True)}"
                )
            awq_metrics = _derive_awq_metrics(awq_stats) if awq_stats else {}
            benchmark_stats = _collect_runtime_stats(llm, phase="benchmark")
            profile_stats = dict(benchmark_stats.get("profile") or {})
            return {
                "skipped": 0.0,
                "timed_out": 1.0,
                "total_tokens": 0.0,
                "wall_time_sec": float(spec.max_run_seconds),
                "aggregate_tps": 0.0,
                "ttft_p50_ms": float("nan"),
                "ttft_p95_ms": float("nan"),
                "e2e_p50_ms": float("nan"),
                "e2e_p95_ms": float("nan"),
                "prefill_p50_ms": float("nan"),
                "prefill_p95_ms": float("nan"),
                "decode_p50_ms": float("nan"),
                "decode_p95_ms": float("nan"),
                "prompt_tokens_total": 0.0,
                "prefill_tps_aggregate": 0.0,
                "prefill_tps_p50": float("nan"),
                "prefill_tps_p95": float("nan"),
                "decode_ms_total": 0.0,
                "decode_tokens_total": 0.0,
                "decode_tps_aggregate": 0.0,
                "decode_tps_p50": float("nan"),
                "decode_tps_p95": float("nan"),
                "stream_output_events_total": 0.0,
                "stream_progressive_token_events_total": 0.0,
                "stream_token_events_total": 0.0,
                "stream_progressive_visible_ratio": 0.0,
                "awq_runtime_stats": awq_stats,
                "awq_metrics": awq_metrics,
                "workload": {
                    "kind": workload,
                    "multimodal_images_per_request": (
                        1
                        if workload == "multimodal_single"
                        else max(2, int(multimodal_images_per_request))
                    ),
                    "multimodal_image_size": max(2, int(multimodal_image_size)),
                },
                "profile": profile_stats,
                "runtime_stats": {
                    **runtime_stats_by_phase,
                    "benchmark": benchmark_stats,
                    "phase_diffs": _derive_runtime_phase_diffs(
                        {
                            **runtime_stats_by_phase,
                            "benchmark": benchmark_stats,
                        }
                    ),
                },
                "warmup_trace": warmup_trace,
            }
        wall_end = time.perf_counter()
        runtime_stats_by_phase["benchmark"] = _collect_runtime_stats(
            llm, phase="benchmark"
        )
        profile_stats = dict(runtime_stats_by_phase["benchmark"].get("profile") or {})
        runtime_stats_by_phase["phase_diffs"] = _derive_runtime_phase_diffs(
            runtime_stats_by_phase
        )

        wall_sec = max(1e-6, wall_end - wall_start)
        total_tokens = int(sum(r["tokens"] for r in request_results))
        aggregate_tps = float(total_tokens) / wall_sec
        ttft_list = _finite_values([r["ttft_ms"] for r in request_results])
        e2e_list = _finite_values([r["e2e_ms"] for r in request_results])
        decode_ms_list = _finite_values([r["decode_ms"] for r in request_results])
        decode_tps_list = _finite_values([r["decode_tps"] for r in request_results])
        prefill_tps_list = _finite_values([r["prefill_tps"] for r in request_results])
        progressive_visible_list = [
            float(r.get("stream_progressive_visible", 0.0) or 0.0)
            for r in request_results
        ]
        prompt_tokens_total = float(sum(r["prompt_tokens"] for r in request_results))
        decode_tokens_total = float(sum(r["decode_tokens"] for r in request_results))
        prefill_ms_total = float(sum(ttft_list)) if ttft_list else 0.0
        decode_ms_total = float(sum(decode_ms_list)) if decode_ms_list else 0.0
        prefill_tps_aggregate = (
            (prompt_tokens_total * 1000.0 / prefill_ms_total)
            if prompt_tokens_total > 0.0 and prefill_ms_total > 0.0
            else 0.0
        )
        decode_tps_aggregate = (
            (decode_tokens_total * 1000.0 / decode_ms_total)
            if decode_tokens_total > 0.0 and decode_ms_total > 0.0
            else 0.0
        )
        awq_stats = {}
        awq_metrics: dict[str, float] = {}
        if spec.quant in ("awq", "compressed-tensors"):
            from vllm.model_executor.layers.quantization.tensor import (
                get_awq_runtime_stats,
            )

            awq_stats = get_awq_runtime_stats()
            awq_metrics = _derive_awq_metrics(awq_stats)

        result = {
            "skipped": 0.0,
            "timed_out": 0.0,
            "total_tokens": float(total_tokens),
            "wall_time_sec": wall_sec,
            "aggregate_tps": aggregate_tps,
            "ttft_p50_ms": median(ttft_list) if ttft_list else float("nan"),
            "ttft_p95_ms": _p95(ttft_list) if ttft_list else float("nan"),
            "e2e_p50_ms": median(e2e_list) if e2e_list else float("nan"),
            "e2e_p95_ms": _p95(e2e_list) if e2e_list else float("nan"),
            "prefill_p50_ms": median(ttft_list) if ttft_list else float("nan"),
            "prefill_p95_ms": _p95(ttft_list) if ttft_list else float("nan"),
            "decode_p50_ms": median(decode_ms_list) if decode_ms_list else float("nan"),
            "decode_p95_ms": _p95(decode_ms_list) if decode_ms_list else float("nan"),
            "prompt_tokens_total": prompt_tokens_total,
            "prefill_tps_aggregate": prefill_tps_aggregate,
            "prefill_tps_p50": median(prefill_tps_list)
            if prefill_tps_list
            else float("nan"),
            "prefill_tps_p95": _p95(prefill_tps_list)
            if prefill_tps_list
            else float("nan"),
            "decode_ms_total": decode_ms_total,
            "decode_tokens_total": decode_tokens_total,
            "decode_tps_aggregate": decode_tps_aggregate,
            "decode_tps_p50": median(decode_tps_list)
            if decode_tps_list
            else float("nan"),
            "decode_tps_p95": _p95(decode_tps_list)
            if decode_tps_list
            else float("nan"),
            "stream_output_events_total": float(
                sum(r.get("stream_output_events", 0.0) for r in request_results)
            ),
            "stream_progressive_token_events_total": float(
                sum(
                    r.get("stream_progressive_token_events", 0.0)
                    for r in request_results
                )
            ),
            "stream_token_events_total": float(
                sum(r.get("stream_token_events", 0.0) for r in request_results)
            ),
            "stream_progressive_visible_ratio": (
                (sum(progressive_visible_list) / len(progressive_visible_list))
                if progressive_visible_list
                else 0.0
            ),
            "awq_runtime_stats": awq_stats,
            "awq_metrics": awq_metrics,
            "workload": {
                "kind": workload,
                "multimodal_images_per_request": (
                    1
                    if workload == "multimodal_single"
                    else max(2, int(multimodal_images_per_request))
                ),
                "multimodal_image_size": max(2, int(multimodal_image_size)),
            },
            "stable_env": dict(spec.stable_env),
            "profile": profile_stats,
            "runtime_stats": runtime_stats_by_phase,
            "warmup_trace": warmup_trace,
        }

        print(
            "  RESULT: "
            f"tokens/s={_fmt_float(result['aggregate_tps'], '.2f')}, "
            f"TTFT p50/p95={_fmt_float(result['ttft_p50_ms'], '.1f')}/{_fmt_float(result['ttft_p95_ms'], '.1f')} ms, "
            f"E2E p50/p95={_fmt_float(result['e2e_p50_ms'], '.1f')}/{_fmt_float(result['e2e_p95_ms'], '.1f')} ms, "
            f"Prefill TPS(agg)={_fmt_float(result['prefill_tps_aggregate'], '.2f')}, "
            f"Prefill TPS p50={_fmt_float(result['prefill_tps_p50'], '.2f')}, "
            f"Decode p50/p95={_fmt_float(result['decode_p50_ms'], '.1f')}/{_fmt_float(result['decode_p95_ms'], '.1f')} ms, "
            f"Decode TPS(agg)={_fmt_float(result['decode_tps_aggregate'], '.2f')}, "
            f"Decode TPS p50={_fmt_float(result['decode_tps_p50'], '.2f')}, "
            f"stream_visible={_fmt_float(result['stream_progressive_visible_ratio'] * 100.0, '.1f')}%"
        )
        print("  " + _format_profile_summary(profile_stats))
        if spec.concurrent_reqs > 1:
            print(
                "  [Note] Decode TPS(agg)=decode_tokens_total/decode_ms_total. "
                "Per-request decode TPS can still be noisy when decode windows are short."
            )
        for line in _format_runtime_phase_diff_summary(
            runtime_stats_by_phase.get("phase_diffs", {})
        ):
            print(line)
        for line in _format_runtime_snapshot_summary(
            runtime_stats_by_phase.get("benchmark", {})
        ):
            print(line)
        if awq_stats:
            print(
                f"  AWQ_STATS: {json.dumps(awq_stats, ensure_ascii=True, sort_keys=True)}"
            )
        return result
    finally:
        if llm is not None:
            llm.shutdown()
        if image_tmpdir is not None:
            image_tmpdir.cleanup()
        try:
            from vllm.model_executor.layers.quantization.tensor import (
                clear_global_weight_cache,
            )

            clear_global_weight_cache()
        except Exception:
            pass
        _restore_env(old_env)
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _safe_metric(value: object) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _deepseek_v4_flash_decode_target_warning(
    summary: dict[str, dict[str, Any]],
) -> dict[str, object] | None:
    model_key = "deepseek_v4_flash_q2_gguf"
    result = summary.get(model_key)
    if not isinstance(result, dict):
        return None
    if result.get("skipped", 0.0) == 1.0 or result.get("timed_out", 0.0) == 1.0:
        return None
    target = 1.0
    current = _safe_metric(result.get("decode_tps_p50"))
    if current is not None and current >= target:
        return None
    return {
        "model": model_key,
        "metric": "decode_tps_p50",
        "kind": (
            "throughput_target" if current is not None else "throughput_target_missing"
        ),
        "current": current if current is not None else float("nan"),
        "baseline": target,
        "ratio": current / target if current is not None else float("nan"),
        "threshold": target,
    }


def _deepseek_smoke_payload_to_benchmark_result(
    spec: ModelSpec,
    payload: dict[str, Any],
    *,
    wall_sec: float,
) -> dict[str, Any]:
    runs = payload.get("runs", [])
    if not isinstance(runs, list):
        runs = []
    elapsed_ms_values = _finite_values(
        [float(run.get("elapsed_ms", 0.0)) for run in runs if isinstance(run, dict)]
    )
    tps_values = _finite_values(
        [
            float(
                run.get(
                    "decode_tps_steady_state",
                    run.get("tokens_per_second", 0.0),
                )
            )
            for run in runs
            if isinstance(run, dict)
        ]
    )
    explicit_decode_tokens = _finite_values(
        [
            float(run.get("decode_tokens_total", 0.0))
            for run in runs
            if isinstance(run, dict) and "decode_tokens_total" in run
        ]
    )
    explicit_decode_ms = _finite_values(
        [
            float(run.get("decode_ms_total", 0.0))
            for run in runs
            if isinstance(run, dict) and "decode_ms_total" in run
        ]
    )
    explicit_prefill_tokens = _finite_values(
        [
            float(run.get("prefill_tokens_total", 0.0))
            for run in runs
            if isinstance(run, dict) and "prefill_tokens_total" in run
        ]
    )
    explicit_prefill_ms = _finite_values(
        [
            float(run.get("prefill_ms_total", 0.0))
            for run in runs
            if isinstance(run, dict) and "prefill_ms_total" in run
        ]
    )
    prefill_tps_values = _finite_values(
        [
            float(run.get("prefill_tps", 0.0))
            for run in runs
            if isinstance(run, dict) and "prefill_tps" in run
        ]
    )
    max_tokens = float(payload.get("max_tokens", spec.max_new_tokens) or 0.0)
    repeat = float(payload.get("repeat", len(runs) or 1) or 1.0)
    decode_tokens_total = (
        float(sum(explicit_decode_tokens))
        if explicit_decode_tokens
        else max_tokens * repeat
    )
    decode_ms_total = (
        float(sum(explicit_decode_ms))
        if explicit_decode_ms
        else float(sum(elapsed_ms_values))
        if elapsed_ms_values
        else 0.0
    )
    prompt_length = float(
        payload.get("prompt_length", spec.prompt_tokens_target) or 0.0
    )
    prefill_tokens_total = (
        float(sum(explicit_prefill_tokens))
        if explicit_prefill_tokens
        else max(prompt_length - 1.0, 0.0) * repeat
    )
    prefill_ms_total = float(sum(explicit_prefill_ms)) if explicit_prefill_ms else 0.0
    decode_tps_aggregate = (
        decode_tokens_total * 1000.0 / decode_ms_total
        if decode_tokens_total > 0.0 and decode_ms_total > 0.0
        else 0.0
    )
    prefill_tps_aggregate = (
        prefill_tokens_total * 1000.0 / prefill_ms_total
        if prefill_tokens_total > 0.0 and prefill_ms_total > 0.0
        else 0.0
    )
    aggregate_tps = (
        decode_tokens_total / wall_sec
        if decode_tokens_total > 0.0 and wall_sec > 0.0
        else 0.0
    )
    gpu_backend = payload.get("gpu_backend", {})
    gpu_staging = payload.get("gpu_staging", {})
    runtime_budget = payload.get("runtime_budget", {})
    return {
        "skipped": 0.0,
        "timed_out": 0.0,
        "total_tokens": decode_tokens_total,
        "wall_time_sec": wall_sec,
        "aggregate_tps": aggregate_tps,
        "ttft_p50_ms": (
            median(elapsed_ms_values) if elapsed_ms_values else float("nan")
        ),
        "ttft_p95_ms": (_p95(elapsed_ms_values) if elapsed_ms_values else float("nan")),
        "e2e_p50_ms": (
            median(elapsed_ms_values) if elapsed_ms_values else float("nan")
        ),
        "e2e_p95_ms": (_p95(elapsed_ms_values) if elapsed_ms_values else float("nan")),
        "prefill_p50_ms": (
            median(explicit_prefill_ms) if explicit_prefill_ms else float("nan")
        ),
        "prefill_p95_ms": (
            _p95(explicit_prefill_ms) if explicit_prefill_ms else float("nan")
        ),
        "decode_p50_ms": (
            median(elapsed_ms_values) if elapsed_ms_values else float("nan")
        ),
        "decode_p95_ms": (
            _p95(elapsed_ms_values) if elapsed_ms_values else float("nan")
        ),
        "prompt_tokens_total": prefill_tokens_total,
        "prefill_tps_aggregate": prefill_tps_aggregate,
        "prefill_tps_p50": (
            median(prefill_tps_values) if prefill_tps_values else float("nan")
        ),
        "prefill_tps_p95": (
            _p95(prefill_tps_values) if prefill_tps_values else float("nan")
        ),
        "decode_ms_total": decode_ms_total,
        "decode_tokens_total": decode_tokens_total,
        "decode_tps_aggregate": decode_tps_aggregate,
        "decode_tps_p50": median(tps_values) if tps_values else float("nan"),
        "decode_tps_p95": _p95(tps_values) if tps_values else float("nan"),
        "stream_output_events_total": 0.0,
        "stream_progressive_token_events_total": 0.0,
        "stream_token_events_total": 0.0,
        "stream_progressive_visible_ratio": 0.0,
        "awq_runtime_stats": {},
        "awq_metrics": {},
        "workload": {
            "kind": "deepseek_v4_flash_gguf",
            "context_length": int(payload.get("context_length", spec.max_model_len)),
            "prompt_length": int(prompt_length),
            "max_tokens": int(max_tokens),
        },
        "stable_env": dict(spec.stable_env),
        "profile": payload.get("profile", {}),
        "runtime_stats": {
            "deepseek_v4_flash": {
                "gpu_backend": gpu_backend if isinstance(gpu_backend, dict) else {},
                "gpu_staging": gpu_staging if isinstance(gpu_staging, dict) else {},
                "runtime_budget": (
                    runtime_budget if isinstance(runtime_budget, dict) else {}
                ),
                "phase3_metrics": payload.get("phase3_metrics", {}),
                "phase4_metrics": payload.get("phase4_metrics", {}),
                "usable_inference_metrics": payload.get("usable_inference_metrics", {}),
            }
        },
        "warmup_trace": [],
    }


def _run_deepseek_v4_flash_gguf_benchmark(spec: ModelSpec) -> dict[str, Any]:
    print("  [DeepSeek] Running GGUF GPU smoke benchmark.")
    if not os.path.isfile(spec.model_path):
        print("  [Skip] DeepSeek V4 Flash GGUF file not found.")
        return {"skipped": 1.0, "skip_reason": "model_file_not_found"}
    command = [
        sys.executable,
        "tests/tools/run_deepseek_v4_flash_gpu_smoke.py",
        "--model",
        spec.model_path,
        "--context-length",
        str(spec.max_model_len),
        "--max-tokens",
        str(spec.max_new_tokens),
    ]
    command.extend(
        [
            "--prompt-length",
            str(spec.prompt_tokens_target),
            "--warmup-tokens",
            str(spec.max_new_tokens),
            "--repeat",
            "3",
            "--min-steady-decode-tps",
            os.environ.get(
                "FASTINFERENCE_DEEPSEEK_V4_FLASH_MIN_STEADY_DECODE_TPS",
                "1.5",
            ),
            "--full-resident",
        ]
    )
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=spec.max_run_seconds,
        )
        wall_sec = max(1e-6, time.perf_counter() - start)
        if proc.returncode != 0:
            raise RuntimeError(
                "DeepSeek V4 Flash GGUF benchmark failed with "
                f"rc={proc.returncode}: {proc.stderr}"
            )
        payload = json.loads(proc.stdout)
        result = _deepseek_smoke_payload_to_benchmark_result(
            spec,
            payload,
            wall_sec=wall_sec,
        )
        print(
            "  RESULT: "
            f"tokens/s={_fmt_float(result['aggregate_tps'], '.2f')}, "
            f"decode={_fmt_float(result['decode_p50_ms'], '.1f')} ms, "
            f"decode_tps={_fmt_float(result['decode_tps_aggregate'], '.2f')}"
        )
        return result
    except subprocess.TimeoutExpired:
        return {
            "skipped": 0.0,
            "timed_out": 1.0,
            "total_tokens": 0.0,
            "wall_time_sec": float(spec.max_run_seconds),
            "aggregate_tps": 0.0,
        }


def _load_perf_baseline(path: str) -> dict[str, Any]:
    if not path.strip():
        return {}
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    summary = payload.get("summary", payload)
    return dict(summary) if isinstance(summary, dict) else {}


def _benchmark_fingerprint(
    args: argparse.Namespace, model_keys: list[str]
) -> dict[str, Any]:
    """Identify results that may be compared as a performance baseline."""
    device = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "models": model_keys,
        "gpu": device.name if device is not None else None,
        "gpu_arch": getattr(device, "gcnArchName", None)
        if device is not None
        else None,
        "rocm": torch.version.hip,
        "torch": torch.__version__,
        "triton": getattr(torch.version, "triton", None),
        "workload": args.workload,
        "fixed_decode_len": bool(args.fixed_decode_len),
        "warmup_preset": args.warmup_preset,
        "shape": {
            key: value
            for key, value in vars(args).items()
            if any(
                token in key
                for token in (
                    "concurrent",
                    "prompt_tokens",
                    "max_new_tokens",
                    "max_model_len",
                )
            )
            and isinstance(value, (bool, int, float, str, type(None)))
        },
        "runtime_env": {
            key: os.environ.get(key)
            for key in ("FASTINFERENCE_KV_TYPE", "FASTINFERENCE_FUSION_LEVEL")
        },
    }


def _evaluate_perf_regressions(
    summary: dict[str, dict[str, Any]],
    baseline: dict[str, Any],
    *,
    min_tps_ratio: float,
    max_latency_ratio: float,
) -> list[dict[str, object]]:
    if not baseline:
        return []
    warnings: list[dict[str, object]] = []
    throughput_metrics = (
        "aggregate_tps",
        "prefill_tps_aggregate",
        "decode_tps_aggregate",
    )
    latency_metrics = ("ttft_p50_ms", "ttft_p95_ms", "e2e_p50_ms", "e2e_p95_ms")
    min_tps_ratio = max(0.0, float(min_tps_ratio))
    max_latency_ratio = max(1.0, float(max_latency_ratio))
    for model_key, current in summary.items():
        if current.get("skipped", 0.0) == 1.0 or current.get("timed_out", 0.0) == 1.0:
            continue
        base = baseline.get(model_key, {})
        if not isinstance(base, dict):
            continue
        for metric in throughput_metrics:
            cur = _safe_metric(current.get(metric))
            ref = _safe_metric(base.get(metric))
            if cur is None or ref is None or ref <= 0.0:
                continue
            ratio = cur / ref
            if ratio < min_tps_ratio:
                warnings.append(
                    {
                        "model": model_key,
                        "metric": metric,
                        "kind": "throughput_drop",
                        "current": cur,
                        "baseline": ref,
                        "ratio": ratio,
                        "threshold": min_tps_ratio,
                    }
                )
        for metric in latency_metrics:
            cur = _safe_metric(current.get(metric))
            ref = _safe_metric(base.get(metric))
            if cur is None or ref is None or ref <= 0.0:
                continue
            ratio = cur / ref
            if ratio > max_latency_ratio:
                warnings.append(
                    {
                        "model": model_key,
                        "metric": metric,
                        "kind": "latency_increase",
                        "current": cur,
                        "baseline": ref,
                        "ratio": ratio,
                        "threshold": max_latency_ratio,
                    }
                )
    return warnings


def _format_perf_regression_warnings(warnings: list[dict[str, object]]) -> list[str]:
    lines: list[str] = []
    for item in warnings:
        lines.append(
            "PERF WARNING: "
            f"model={item.get('model')} metric={item.get('metric')} "
            f"kind={item.get('kind')} current={_fmt_float(float(item.get('current', float('nan'))), '.3f')} "
            f"baseline={_fmt_float(float(item.get('baseline', float('nan'))), '.3f')} "
            f"ratio={_fmt_float(float(item.get('ratio', float('nan'))), '.3f')} "
            f"threshold={_fmt_float(float(item.get('threshold', float('nan'))), '.3f')}"
        )
    return lines


def _perf_gate_failed(
    regressions: list[dict[str, object]], *, fail_on_regression: bool
) -> bool:
    return bool(fail_on_regression and regressions)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end performance benchmark for Gemma4 12B/26B/31B, Qwen3.5, "
            "TinyLlama, and DeepSeek V4 Flash Q2 GGUF models."
        )
    )
    parser.add_argument(
        "--models",
        type=str,
        default="gemma4_26b_a4b,gemma4_31b_q4,deepseek_v4_flash_q2_gguf",
        help=(
            "Comma-separated model keys. Default: "
            "gemma4_26b_a4b,gemma4_31b_q4,deepseek_v4_flash_q2_gguf. "
            "Available: tinyllama, qwen35_9b_awq, gemma4_12b_awq, gemma4_31b_q4, "
            "gemma4_26b_a4b, deepseek_v4_flash_q2_gguf."
        ),
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional path to save JSON benchmark summary.",
    )
    parser.add_argument(
        "--runtime-stats-out",
        type=str,
        default="",
        help=(
            "Optional path to save runtime stats snapshots only. If unset and --json-out "
            "is provided, runtime stats are included in the main summary JSON."
        ),
    )
    parser.add_argument(
        "--aggressive",
        action="store_true",
        help=(
            "Use full MODEL_SPECS (e.g. BS=8, ~4096-token prefill). On ROCm the default is a "
            "safer profile to avoid HIP launch failures; this flag or FASTINFERENCE_E2E_AGGRESSIVE=1 "
            "restores aggressive settings (needs ample VRAM)."
        ),
    )
    parser.add_argument(
        "--workload",
        type=str,
        default="text",
        choices=("text", "multimodal_single", "multimodal_multi"),
        help="Benchmark workload type. multimodal_* generates real image_url requests.",
    )
    parser.add_argument(
        "--multimodal-images-per-request",
        type=int,
        default=2,
        metavar="N",
        help="Number of images per request for multimodal_multi workload.",
    )
    parser.add_argument(
        "--multimodal-image-size",
        type=int,
        default=8,
        metavar="PX",
        help="Edge size in pixels for generated benchmark images.",
    )
    parser.add_argument(
        "--tinyllama-concurrent",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Override concurrent request count for tinyllama only. "
            "Default from MODEL_SPECS is 8 (with ~4096-token prefill)."
        ),
    )
    parser.add_argument(
        "--qwen9b-concurrent",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Override concurrent request count (batch width) for qwen35_9b_awq only. "
            "Default from MODEL_SPECS is 8 (with ~4096-token prefill; 48GB+ VRAM typical)."
        ),
    )
    parser.add_argument(
        "--gemma31b-concurrent",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Override concurrent request count (batch width) for gemma4_31b_q4 only. "
            "Default from MODEL_SPECS is 2."
        ),
    )
    parser.add_argument(
        "--gemma31b-prompt-tokens",
        type=int,
        default=None,
        metavar="N",
        help="Override prompt token target for gemma4_31b_q4.",
    )
    parser.add_argument(
        "--gemma31b-max-new-tokens",
        type=int,
        default=None,
        metavar="N",
        help="Override max_new_tokens for gemma4_31b_q4.",
    )
    parser.add_argument(
        "--gemma31b-max-model-len",
        type=int,
        default=None,
        metavar="N",
        help="Override max_model_len for gemma4_31b_q4.",
    )
    parser.add_argument(
        "--gemma26b-concurrent",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Override concurrent request count (batch width) for gemma4_26b_a4b only. "
            "Default from MODEL_SPECS is 1."
        ),
    )
    parser.add_argument(
        "--gemma26b-prompt-tokens",
        type=int,
        default=None,
        metavar="N",
        help="Override prompt token target for gemma4_26b_a4b.",
    )
    parser.add_argument(
        "--gemma26b-max-new-tokens",
        type=int,
        default=None,
        metavar="N",
        help="Override max_new_tokens for gemma4_26b_a4b.",
    )
    parser.add_argument(
        "--gemma26b-max-model-len",
        type=int,
        default=None,
        metavar="N",
        help="Override max_model_len for gemma4_26b_a4b.",
    )
    parser.add_argument(
        "--deepseek-concurrent",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Override concurrent request count (batch width) for "
            "deepseek_v4_flash_q2_gguf only. Default from MODEL_SPECS is 1."
        ),
    )
    parser.add_argument(
        "--deepseek-prompt-tokens",
        type=int,
        default=None,
        metavar="N",
        help="Override prompt token target for deepseek_v4_flash_q2_gguf.",
    )
    parser.add_argument(
        "--warmup-prefill-rounds",
        type=int,
        default=1,
        metavar="N",
        help="Warmup prefill rounds before benchmark (default: 1).",
    )
    parser.add_argument(
        "--warmup-preset",
        type=str,
        default="default",
        choices=("default", "off", "cold"),
        help=(
            "Warmup preset. default uses explicit warmup-* args; "
            "off disables warmup; cold adds stronger compile/jitter warmup."
        ),
    )
    parser.add_argument(
        "--warmup-decode-rounds",
        type=int,
        default=1,
        metavar="N",
        help="Warmup decode rounds before benchmark (default: 1).",
    )
    parser.add_argument(
        "--warmup-decode-tokens",
        type=int,
        default=8,
        metavar="N",
        help="max_tokens used by each warmup decode round (default: 8).",
    )
    parser.add_argument(
        "--warmup-burst-rounds",
        type=int,
        default=0,
        metavar="N",
        help="Optional concurrent warmup rounds for first-request jitter control.",
    )
    parser.add_argument(
        "--warmup-burst-concurrency",
        type=int,
        default=0,
        metavar="N",
        help="Concurrency used in warmup burst rounds (0 disables burst warmup).",
    )
    parser.add_argument(
        "--warmup-burst-decode-tokens",
        type=int,
        default=8,
        metavar="N",
        help="max_tokens used by each request in burst warmup rounds.",
    )
    parser.add_argument(
        "--compile-cache-dir",
        type=str,
        default="",
        metavar="PATH",
        help=(
            "Optional compile cache root (sets TRITON_CACHE_DIR and TORCHINDUCTOR_CACHE_DIR). "
            "Can also be set via FASTINFERENCE_BENCH_COMPILE_CACHE_DIR."
        ),
    )
    parser.add_argument(
        "--compile-cache-clear",
        action="store_true",
        help="Clear --compile-cache-dir before run (cold-start style measurement).",
    )
    parser.add_argument(
        "--gemma31b-bucket-enable",
        action="store_true",
        default=True,
        help="Enable default prompt-bucket scheduling policy for Gemma4-31B.",
    )
    parser.add_argument(
        "--gemma31b-bucket-disable",
        action="store_false",
        dest="gemma31b_bucket_enable",
        help="Disable prompt-bucket scheduling policy for Gemma4-31B.",
    )
    parser.add_argument(
        "--gemma31b-bucket-cutoff",
        type=int,
        default=128,
        metavar="N",
        help="Prompt-token cutoff for Gemma4-31B auto bucket routing.",
    )
    parser.add_argument(
        "--gemma31b-bucket-decode-cutoff",
        type=int,
        default=32,
        metavar="N",
        help="Decode-token cutoff used inside the short-prompt bucket for Gemma4-31B.",
    )
    parser.add_argument(
        "--gemma31b-bucket-short-profile",
        type=str,
        default="decode_bias",
        choices=("baseline", "decode_bias", "catchup_prefill"),
        help="Scheduler profile for short prompts (<= cutoff) with longer decode lengths.",
    )
    parser.add_argument(
        "--gemma31b-bucket-short-decode-profile",
        type=str,
        default="catchup_prefill",
        choices=("baseline", "decode_bias", "catchup_prefill"),
        help="Scheduler profile for short prompts (<= cutoff) with decode <= decode-cutoff.",
    )
    parser.add_argument(
        "--gemma31b-bucket-long-profile",
        type=str,
        default="baseline",
        choices=("baseline", "decode_bias", "catchup_prefill"),
        help="Scheduler profile for long prompts (> cutoff).",
    )
    parser.add_argument(
        "--gemma31b-schedule-profile",
        type=str,
        default="auto_bucket",
        choices=("auto_bucket", "baseline", "decode_bias", "catchup_prefill"),
        help=(
            "Gemma4-31B scheduler profile mode. auto_bucket routes by prompt bucket; "
            "other values force one profile."
        ),
    )
    parser.add_argument(
        "--fixed-decode-len",
        action="store_true",
        default=True,
        help=(
            "Force fixed decode length by setting ignore_eos=True and min_tokens=max_new_tokens "
            "(default: enabled)."
        ),
    )
    parser.add_argument(
        "--allow-eos-stop",
        action="store_false",
        dest="fixed_decode_len",
        help="Disable fixed decode length and allow early stop on EOS.",
    )
    parser.add_argument(
        "--reuse-warmup-prefix-cache",
        action="store_true",
        default=False,
        help=(
            "Keep prefix-cache entries created by warmup requests. Default benchmark "
            "runs clear them so measured TTFT/decode match the repository baseline "
            "rather than a warm-cache repeated-prompt scenario."
        ),
    )
    parser.add_argument(
        "--perf-baseline-json",
        type=str,
        default="",
        metavar="PATH",
        help=(
            "Optional prior e2e_full_benchmark JSON. When provided, this run "
            "emits structured warnings for throughput drops or latency increases."
        ),
    )
    parser.add_argument(
        "--perf-warn-min-tps-ratio",
        type=float,
        default=0.85,
        metavar="RATIO",
        help="Warn when current throughput metric is below this fraction of baseline.",
    )
    parser.add_argument(
        "--perf-warn-max-latency-ratio",
        type=float,
        default=1.25,
        metavar="RATIO",
        help="Warn when current latency metric exceeds this multiple of baseline.",
    )
    parser.add_argument(
        "--perf-fail-on-regression",
        action="store_true",
        help="Exit nonzero when --perf-baseline-json detects a regression.",
    )
    parser.add_argument(
        "--model-process-isolation",
        action="store_true",
        default=False,
        help=(
            "Run each requested model in a fresh child process and merge JSON summaries. "
            "Useful on ROCm when sequential large-model runs OOM due to allocator residue."
        ),
    )
    parser.add_argument(
        "--no-model-process-isolation",
        action="store_true",
        default=False,
        help="Disable per-model child-process isolation even when auto-isolation would apply.",
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    aggressive = _is_e2e_aggressive(args)
    warmup_config = _resolve_warmup_config(args)
    perf_baseline = _load_perf_baseline(args.perf_baseline_json)
    compile_cache_env, compile_cache_meta = _resolve_compile_cache_env(args)
    compile_cache_before = _cache_dir_stats(compile_cache_meta.get("cache_dir"))
    compile_cache_after = dict(compile_cache_before)
    if _is_rocm() and not aggressive:
        print(
            "[ROCm] Using conservative E2E defaults "
            f"(concurrent<={_ROCM_E2E_SAFE_CONCURRENT}, prompt<={_ROCM_E2E_SAFE_PROMPT_TOKENS}, "
            f"gpu_memory_utilization<={_ROCM_E2E_SAFE_GPU_MEM_UTIL}) to reduce HIP launch failures."
        )
        print(
            "       For BS=8 / ~4096-ctx: pass --aggressive or set FASTINFERENCE_E2E_AGGRESSIVE=1 "
            "(needs large VRAM; if you still see hipErrorLaunchFailure, try AMD_SERIALIZE_KERNEL=3 to locate)."
        )
    model_keys = [k.strip() for k in args.models.split(",") if k.strip()]
    specs: list[ModelSpec] = []
    resolved_scheduler_policy: dict[str, dict[str, Any]] = {}
    for key in model_keys:
        if key not in MODEL_SPECS:
            raise ValueError(
                f"Unknown model key: {key}. Supported: {', '.join(MODEL_SPECS.keys())}"
            )
        spec = _maybe_apply_rocm_safe_profile(MODEL_SPECS[key], aggressive)
        if key == "tinyllama" and args.tinyllama_concurrent is not None:
            n = int(args.tinyllama_concurrent)
            if n < 1 or n > 64:
                raise ValueError("--tinyllama-concurrent must be between 1 and 64")
            spec = replace(
                spec,
                concurrent_reqs=n,
                max_run_seconds=max(spec.max_run_seconds, 60 * n),
            )
        if key == "qwen35_9b_awq" and args.qwen9b_concurrent is not None:
            n = int(args.qwen9b_concurrent)
            if n < 1 or n > 64:
                raise ValueError("--qwen9b-concurrent must be between 1 and 64")
            spec = replace(
                spec,
                concurrent_reqs=n,
                max_run_seconds=max(spec.max_run_seconds, 60 * n),
            )
        if key == "gemma4_31b_q4" and args.gemma31b_concurrent is not None:
            n = int(args.gemma31b_concurrent)
            if n < 1 or n > 16:
                raise ValueError("--gemma31b-concurrent must be between 1 and 16")
            spec = replace(
                spec,
                concurrent_reqs=n,
                max_run_seconds=max(spec.max_run_seconds, 120 * n),
            )
        if key == "gemma4_31b_q4" and args.gemma31b_prompt_tokens is not None:
            n = int(args.gemma31b_prompt_tokens)
            if n < 8:
                raise ValueError("--gemma31b-prompt-tokens must be >= 8")
            spec = replace(spec, prompt_tokens_target=n)
        if key == "gemma4_31b_q4" and args.gemma31b_max_new_tokens is not None:
            n = int(args.gemma31b_max_new_tokens)
            if n < 1:
                raise ValueError("--gemma31b-max-new-tokens must be >= 1")
            spec = replace(spec, max_new_tokens=n)
        if key == "gemma4_31b_q4" and args.gemma31b_max_model_len is not None:
            n = int(args.gemma31b_max_model_len)
            if n < 64:
                raise ValueError("--gemma31b-max-model-len must be >= 64")
            new_env = dict(spec.stable_env)
            new_env["FASTINFERENCE_KV_MAX_MODEL_LEN"] = str(n)
            spec = replace(spec, max_model_len=n, stable_env=new_env)
        if key == "gemma4_31b_q4":
            spec, policy = _resolve_gemma31b_bucket_policy(spec, args)
            resolved_scheduler_policy[key] = policy
        if key == "gemma4_26b_a4b" and args.gemma26b_concurrent is not None:
            n = int(args.gemma26b_concurrent)
            if n < 1 or n > 16:
                raise ValueError("--gemma26b-concurrent must be between 1 and 16")
            spec = replace(
                spec,
                concurrent_reqs=n,
                max_run_seconds=max(spec.max_run_seconds, 120 * n),
            )
        if key == "gemma4_26b_a4b" and args.gemma26b_prompt_tokens is not None:
            n = int(args.gemma26b_prompt_tokens)
            if n < 8:
                raise ValueError("--gemma26b-prompt-tokens must be >= 8")
            spec = replace(spec, prompt_tokens_target=n)
        if key == "gemma4_26b_a4b" and args.gemma26b_max_new_tokens is not None:
            n = int(args.gemma26b_max_new_tokens)
            if n < 1:
                raise ValueError("--gemma26b-max-new-tokens must be >= 1")
            spec = replace(spec, max_new_tokens=n)
        if key == "gemma4_26b_a4b" and args.gemma26b_max_model_len is not None:
            n = int(args.gemma26b_max_model_len)
            if n < 64:
                raise ValueError("--gemma26b-max-model-len must be >= 64")
            new_env = dict(spec.stable_env)
            new_env["FASTINFERENCE_KV_MAX_MODEL_LEN"] = str(n)
            spec = replace(spec, max_model_len=n, stable_env=new_env)
        if key == "deepseek_v4_flash_q2_gguf" and args.deepseek_concurrent is not None:
            n = int(args.deepseek_concurrent)
            if n < 1 or n > 16:
                raise ValueError("--deepseek-concurrent must be between 1 and 16")
            spec = replace(
                spec,
                concurrent_reqs=n,
                max_run_seconds=max(spec.max_run_seconds, 120 * n),
            )
        if (
            key == "deepseek_v4_flash_q2_gguf"
            and args.deepseek_prompt_tokens is not None
        ):
            n = int(args.deepseek_prompt_tokens)
            if n < 1:
                raise ValueError("--deepseek-prompt-tokens must be >= 1")
            spec = replace(spec, prompt_tokens_target=n)
        specs.append(spec)

    print("=" * 72)
    print("FASTINFERENCE END-TO-END PERFORMANCE REGRESSION")
    print("Targets: " + _format_targets(specs))
    print(
        "Warmup: "
        f"preset={args.warmup_preset}, "
        f"prefill={warmup_config.prefill_rounds}, "
        f"decode={warmup_config.decode_rounds}x{warmup_config.decode_tokens}, "
        f"burst={warmup_config.burst_rounds}x{warmup_config.burst_concurrency}"
    )
    print(
        "Decode mode: "
        f"{'fixed_len(ignore_eos,min_tokens=max_new_tokens)' if args.fixed_decode_len else 'allow_eos_stop'}"
    )
    if compile_cache_meta.get("enabled"):
        print(
            "Compile cache: "
            f"dir={compile_cache_meta['cache_dir']} "
            f"(files={compile_cache_before['files']}, bytes={compile_cache_before['bytes']})"
        )
    gemma31_policy = resolved_scheduler_policy.get("gemma4_31b_q4")
    if gemma31_policy is not None:
        print(
            "Gemma4-31B scheduler: "
            f"mode={gemma31_policy['mode']} profile={gemma31_policy['selected_profile']} "
            f"bucket={'on' if gemma31_policy['enabled'] else 'off'} "
            f"cutoff={gemma31_policy['cutoff_prompt_tokens']} "
            f"prompt={gemma31_policy['prompt_tokens_target']}"
        )
    print("=" * 72)

    summary: dict[str, dict[str, Any]] = {}
    runtime_stats_summary: dict[str, dict[str, object]] = {}
    use_isolation = _should_use_model_process_isolation(args, model_keys)
    if use_isolation:
        print(
            "[ROCm] Enabling per-model process isolation for this run to avoid "
            "sequential large-model OOM/fragmentation."
        )
        summary, runtime_stats_summary = _run_isolated_model_benchmarks(
            model_keys,
            compile_cache_env,
        )
    else:
        global_old_env = _apply_temp_env(compile_cache_env)
        try:
            for spec in specs:
                summary[spec.key] = await run_benchmark(
                    spec,
                    workload=args.workload,
                    multimodal_images_per_request=args.multimodal_images_per_request,
                    multimodal_image_size=args.multimodal_image_size,
                    warmup_config=warmup_config,
                    fixed_decode_len=bool(args.fixed_decode_len),
                    clear_prefix_cache_after_warmup=_clear_prefix_cache_after_warmup(
                        args
                    ),
                )
                runtime_stats_summary[spec.key] = dict(
                    summary[spec.key].get("runtime_stats", {})
                )
        finally:
            _restore_env(global_old_env)
    compile_cache_after = _cache_dir_stats(compile_cache_meta.get("cache_dir"))
    compile_cache_delta = {
        "files": int(compile_cache_after["files"]) - int(compile_cache_before["files"]),
        "bytes": int(compile_cache_after["bytes"]) - int(compile_cache_before["bytes"]),
    }

    print("\n" + "-" * 72)
    print("PERF SUMMARY")
    print(
        "(decode_tps_agg = decode_tokens_total / decode_ms_total; decode_tps_p50 may be n/a "
        "when token streaming is not progressively visible)"
    )
    print("-" * 72)
    for key in model_keys:
        r = summary[key]
        if r.get("skipped", 0.0) == 1.0:
            print(f"{key:16} | skipped")
            continue
        if r.get("timed_out", 0.0) == 1.0:
            print(f"{key:16} | timeout")
            profile_stats = dict(r.get("profile") or {})
            print(_format_profile_summary(profile_stats))
            continue
        print(
            f"{key:16} | tps={_fmt_float(r['aggregate_tps'], '.2f')} | "
            f"ttft_p50={_fmt_float(r['ttft_p50_ms'], '.1f')}ms | ttft_p95={_fmt_float(r['ttft_p95_ms'], '.1f')}ms | "
            f"prefill_tps_agg={_fmt_float(r.get('prefill_tps_aggregate', float('nan')), '.2f')} | "
            f"e2e_p50={_fmt_float(r['e2e_p50_ms'], '.1f')}ms | e2e_p95={_fmt_float(r['e2e_p95_ms'], '.1f')}ms | "
            f"decode_tps_agg={_fmt_float(r['decode_tps_aggregate'], '.2f')} | "
            f"decode_tps_p50={_fmt_float(r['decode_tps_p50'], '.2f')} | "
            f"stream_visible={_fmt_float(float(r.get('stream_progressive_visible_ratio', 0.0) or 0.0) * 100.0, '.1f')}% | "
            f"workload={r.get('workload', {}).get('kind', args.workload)}"
        )
        profile_stats = dict(r.get("profile") or {})
        print(_format_profile_summary(profile_stats))
    perf_regression_warnings = _evaluate_perf_regressions(
        summary,
        perf_baseline,
        min_tps_ratio=float(args.perf_warn_min_tps_ratio),
        max_latency_ratio=float(args.perf_warn_max_latency_ratio),
    )
    for line in _format_perf_regression_warnings(perf_regression_warnings):
        print(line)
    perf_target_warnings: list[dict[str, object]] = []
    deepseek_target_warning = _deepseek_v4_flash_decode_target_warning(summary)
    if deepseek_target_warning is not None:
        perf_target_warnings.append(deepseek_target_warning)
    for line in _format_perf_regression_warnings(perf_target_warnings):
        print(line)

    if compile_cache_meta.get("enabled"):
        print(
            "compile_cache      | "
            f"files={compile_cache_after['files']} (delta={compile_cache_delta['files']:+d}) | "
            f"bytes={compile_cache_after['bytes']} (delta={compile_cache_delta['bytes']:+d})"
        )

    if args.json_out:
        payload = {
            "fingerprint": _benchmark_fingerprint(args, model_keys),
            "models": model_keys,
            "workload": {
                "kind": args.workload,
                "multimodal_images_per_request": args.multimodal_images_per_request,
                "multimodal_image_size": args.multimodal_image_size,
            },
            "warmup": {
                "preset": args.warmup_preset,
                "prefill_rounds": warmup_config.prefill_rounds,
                "decode_rounds": warmup_config.decode_rounds,
                "decode_tokens": warmup_config.decode_tokens,
                "burst_rounds": warmup_config.burst_rounds,
                "burst_concurrency": warmup_config.burst_concurrency,
                "burst_decode_tokens": warmup_config.burst_decode_tokens,
            },
            "decode_mode": {
                "fixed_decode_len": bool(args.fixed_decode_len),
            },
            "compile_cache": {
                **compile_cache_meta,
                "before": compile_cache_before,
                "after": compile_cache_after,
                "delta": compile_cache_delta,
            },
            "resolved_scheduler_policy": resolved_scheduler_policy,
            "summary": summary,
            "runtime_stats": runtime_stats_summary,
            "perf_regressions": perf_regression_warnings,
            "perf_target_warnings": perf_target_warnings,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
        print(f"\nJSON summary written to: {args.json_out}")

    if args.runtime_stats_out:
        payload = {
            "models": model_keys,
            "workload": {
                "kind": args.workload,
                "multimodal_images_per_request": args.multimodal_images_per_request,
                "multimodal_image_size": args.multimodal_image_size,
            },
            "warmup": {
                "preset": args.warmup_preset,
                "prefill_rounds": warmup_config.prefill_rounds,
                "decode_rounds": warmup_config.decode_rounds,
                "decode_tokens": warmup_config.decode_tokens,
                "burst_rounds": warmup_config.burst_rounds,
                "burst_concurrency": warmup_config.burst_concurrency,
                "burst_decode_tokens": warmup_config.burst_decode_tokens,
            },
            "decode_mode": {
                "fixed_decode_len": bool(args.fixed_decode_len),
            },
            "compile_cache": {
                **compile_cache_meta,
                "before": compile_cache_before,
                "after": compile_cache_after,
                "delta": compile_cache_delta,
            },
            "resolved_scheduler_policy": resolved_scheduler_policy,
            "runtime_stats": runtime_stats_summary,
            "perf_regressions": perf_regression_warnings,
            "perf_target_warnings": perf_target_warnings,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(args.runtime_stats_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
        print(f"Runtime stats written to: {args.runtime_stats_out}")

    if _perf_gate_failed(
        perf_regression_warnings,
        fail_on_regression=bool(args.perf_fail_on_regression),
    ):
        raise SystemExit("performance regression gate failed")


if __name__ == "__main__":
    asyncio.run(main())
