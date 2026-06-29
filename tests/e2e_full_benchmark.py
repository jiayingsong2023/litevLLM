# SPDX-License-Identifier: Apache-2.0
import argparse
import asyncio
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
from typing import Any, Dict, List, Optional

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
    stable_env: Dict[str, str]


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

_GEMMA4_31B_RECOMMENDED_ENV: dict[str, str] = {
    "FASTINFERENCE_KV_TYPE": "turbo_int4",
    "FASTINFERENCE_FUSION_LEVEL": "2",
    "FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS": "1",
    "FASTINFERENCE_KV_MAX_MODEL_LEN": "512",
    "FASTINFERENCE_GEMMA4_DENSE_DOWN_PROJ": "1",
    "FASTINFERENCE_AWQ_DECODE_GEMV": "1",
    "FASTINFERENCE_AWQ_GROUP32_GEMV_ALL": "1",
    "FASTINFERENCE_AWQ_FUSED_GATE_UP": "1",
    "FASTINFERENCE_GPU_GREEDY_SAMPLING": "1",
    "FASTINFERENCE_GPU_GREEDY_MAX_TOKENS_ONLY": "1",
    "FASTINFERENCE_GPU_GREEDY_BYPASS_CPU_POLICIES": "1",
}

_GEMMA4_26B_RECOMMENDED_ENV: dict[str, str] = {
    "FASTINFERENCE_KV_TYPE": "turbo_int4",
    "FASTINFERENCE_FUSION_LEVEL": "2",
    "FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS": "1",
    "FASTINFERENCE_KV_MAX_MODEL_LEN": "512",
    "FASTINFERENCE_AWQ_DECODE_GEMV": "1",
    "FASTINFERENCE_AWQ_FUSED_GATE_UP": "1",
    "FASTINFERENCE_GEMMA4_MOE_EXPERT_CACHE_SIZE": "32",
    "FASTINFERENCE_GEMMA4_MOE_COMPUTE_DTYPE": "auto",
    "FASTINFERENCE_GEMMA4_MOE_INT4_KERNEL": "1",
    "FASTINFERENCE_GPU_GREEDY_SAMPLING": "1",
    "FASTINFERENCE_GPU_GREEDY_MAX_TOKENS_ONLY": "1",
    "FASTINFERENCE_GPU_GREEDY_BYPASS_CPU_POLICIES": "1",
}

_DEEPSEEK_V4_FLASH_GGUF_PATH = (
    "models/DeepSeek-V4-Flash-ds4/"
    "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
)


# KV cache default: TurboQuant INT4 (FASTINFERENCE_KV_TYPE=turbo_int4).
MODEL_SPECS: Dict[str, ModelSpec] = {
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
        stable_env=dict(_GEMMA4_31B_RECOMMENDED_ENV),
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
        stable_env=dict(_GEMMA4_26B_RECOMMENDED_ENV),
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
        prompt_tokens_target=4096,
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
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, object]]]:
    summary: Dict[str, Dict[str, Any]] = {}
    runtime_stats_summary: Dict[str, Dict[str, object]] = {}
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
        with open(child_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        try:
            os.unlink(child_json)
        except OSError:
            pass
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
        with open(config_path, "r", encoding="utf-8") as f:
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


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(0.95 * (len(ordered) - 1))
    return ordered[idx]


def _finite_values(values: List[float]) -> List[float]:
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


def _apply_temp_env(env_map: Dict[str, str]) -> Dict[str, Optional[str]]:
    old_env: Dict[str, Optional[str]] = {}
    for key, value in env_map.items():
        old_env[key] = os.environ.get(key)
        os.environ[key] = value
    return old_env


def _restore_env(old_env: Dict[str, Optional[str]]) -> None:
    for key, value in old_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _collect_runtime_stats(
    llm: AsyncLLM,
    *,
    phase: str,
) -> Dict[str, object]:
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


def _derive_runtime_metrics(snapshot: Dict[str, object]) -> Dict[str, object]:
    observer = snapshot.get("observer", {})
    backend = snapshot.get("backend", {})
    lora_runtime = snapshot.get("lora", {})
    async_driver = snapshot.get("async_driver", {})
    if not isinstance(observer, dict) or not isinstance(backend, dict):
        return {}

    observer_prefix = observer.get("prefix_cache", {})
    backend_prefix = backend.get("prefix_cache", {})
    observer_preemption = observer.get("preemption", {})
    observer_fairness = observer.get("fairness", {})
    observer_lora = observer.get("lora", {})
    observer_multimodal = observer.get("multimodal", {})
    backend_multimodal = backend.get("multimodal", {})
    if not isinstance(observer_prefix, dict):
        observer_prefix = {}
    if not isinstance(backend_prefix, dict):
        backend_prefix = {}
    if not isinstance(observer_preemption, dict):
        observer_preemption = {}
    if not isinstance(observer_fairness, dict):
        observer_fairness = {}
    if not isinstance(observer_lora, dict):
        observer_lora = {}
    if not isinstance(observer_multimodal, dict):
        observer_multimodal = {}
    if not isinstance(backend_multimodal, dict):
        backend_multimodal = {}
    if not isinstance(lora_runtime, dict):
        lora_runtime = {}
    if not isinstance(async_driver, dict):
        async_driver = {}

    request_count = float(observer_prefix.get("events", 0) or 0)
    step_count = float(observer.get("step_count", 0) or 0)
    async_steps = float(async_driver.get("steps", 0) or 0)
    async_backpressure_sleeps = float(async_driver.get("backpressure_sleeps", 0) or 0)
    async_idle_waits = float(async_driver.get("idle_waits", 0) or 0)
    async_background_errors = float(async_driver.get("background_errors", 0) or 0)
    async_min_step_interval_s = float(
        async_driver.get("min_step_interval_s", 0.0) or 0.0
    )
    admitted_requests = float(observer.get("admitted_requests", 0) or 0)
    materialized_hits = float(backend.get("prefix_cache_materialized_hits", 0) or 0)
    materialized_saved = float(
        backend.get("prefix_cache_materialized_saved_prefill_tokens", 0) or 0
    )
    lookup_comparisons = float(backend_prefix.get("lookup_comparisons", 0) or 0)
    lookup_candidates_total = float(
        backend_prefix.get("lookup_candidates_total", 0) or 0
    )
    preempted_steps = float(observer_preemption.get("preempted_steps", 0) or 0)
    preempted_prefills = float(
        observer_preemption.get("preempted_prefill_requests", 0) or 0
    )
    preempted_multimodal_prefills = float(
        observer_preemption.get("preempted_multimodal_prefill_requests", 0) or 0
    )
    protected_multimodal_prefix_steps = float(
        observer_preemption.get("protected_multimodal_prefix_steps", 0) or 0
    )
    protected_multimodal_prefix_prefills = float(
        observer_preemption.get("protected_multimodal_prefix_prefill_requests", 0) or 0
    )
    starvation_protected_steps = float(
        observer_fairness.get("starvation_protected_steps", 0) or 0
    )
    fairness_guardrail_steps = float(
        observer_fairness.get("fairness_guardrail_triggered_steps", 0) or 0
    )
    per_class_p95_wait = observer_fairness.get("per_class_p95_queue_wait_s", {})
    if not isinstance(per_class_p95_wait, dict):
        per_class_p95_wait = {}
    lora_adapters = lora_runtime.get("adapters", {})
    if not isinstance(lora_adapters, dict):
        lora_adapters = {}
    lora_prefill_adapters = observer_lora.get("prefill_adapters", {})
    lora_decode_adapters = observer_lora.get("decode_adapters", {})
    lora_backlog_adapters = observer_lora.get("backlog_adapters", {})
    if not isinstance(lora_prefill_adapters, dict):
        lora_prefill_adapters = {}
    if not isinstance(lora_decode_adapters, dict):
        lora_decode_adapters = {}
    if not isinstance(lora_backlog_adapters, dict):
        lora_backlog_adapters = {}
    lora_prefill_steps = float(observer_lora.get("prefill_step_count", 0) or 0)
    lora_decode_steps = float(observer_lora.get("decode_step_count", 0) or 0)
    admitted_adapter_share = _normalized_share_map(
        observer_lora.get("admitted_adapters", {})
    )
    backlog_adapter_share = _normalized_share_map(
        observer_lora.get("backlog_adapters", {})
    )
    adapter_fairness_gap = _share_gap_map(admitted_adapter_share, backlog_adapter_share)

    return {
        "async_driver": {
            "steps": async_steps,
            "backpressure_sleeps": async_backpressure_sleeps,
            "idle_waits": async_idle_waits,
            "background_errors": async_background_errors,
            "min_step_interval_s": async_min_step_interval_s,
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
        "preemption": {
            "step_count": step_count,
            "preempted_steps": preempted_steps,
            "preempted_prefill_requests": preempted_prefills,
            "preempted_multimodal_prefill_requests": preempted_multimodal_prefills,
            "protected_multimodal_prefix_steps": protected_multimodal_prefix_steps,
            "protected_multimodal_prefix_prefill_requests": (
                protected_multimodal_prefix_prefills
            ),
            "preempted_step_rate": (
                preempted_steps / step_count if step_count else 0.0
            ),
            "preempted_prefill_requests_per_step": (
                preempted_prefills / step_count if step_count else 0.0
            ),
            "preempted_multimodal_prefill_requests_per_step": (
                preempted_multimodal_prefills / step_count if step_count else 0.0
            ),
            "protected_multimodal_prefix_step_rate": (
                protected_multimodal_prefix_steps / step_count if step_count else 0.0
            ),
            "protected_multimodal_prefix_prefill_requests_per_step": (
                protected_multimodal_prefix_prefills / step_count if step_count else 0.0
            ),
        },
        "fairness": {
            "step_count": step_count,
            "admitted_requests": admitted_requests,
            "starvation_protected_steps": starvation_protected_steps,
            "fairness_guardrail_triggered_steps": fairness_guardrail_steps,
            "starvation_protected_step_rate": (
                starvation_protected_steps / step_count if step_count else 0.0
            ),
            "fairness_guardrail_triggered_step_rate": (
                fairness_guardrail_steps / step_count if step_count else 0.0
            ),
            "avg_admitted_queue_wait_s": float(
                observer_fairness.get("avg_admitted_queue_wait_s", 0.0) or 0.0
            ),
            "p95_queue_wait_s": float(
                observer_fairness.get("p95_queue_wait_s", 0.0) or 0.0
            ),
            "max_queue_wait_s": float(
                observer_fairness.get("max_queue_wait_s", 0.0) or 0.0
            ),
            "per_class_p95_queue_wait_s": {
                str(key): float(value or 0.0)
                for key, value in per_class_p95_wait.items()
            },
        },
        "lora": {
            "registered_adapters": float(
                lora_runtime.get("registered_adapters", 0) or 0
            ),
            "active_adapters": float(lora_runtime.get("active_adapters", 0) or 0),
            "active_requests": float(lora_runtime.get("active_requests", 0) or 0),
            "total_routed_requests": float(
                lora_runtime.get("total_routed_requests", 0) or 0
            ),
            "adapter_count_observed": float(len(lora_adapters)),
            "mixed_lora_prefill_steps": float(
                observer_lora.get("mixed_lora_prefill_steps", 0) or 0
            ),
            "mixed_lora_decode_steps": float(
                observer_lora.get("mixed_lora_decode_steps", 0) or 0
            ),
            "admit_relaxed_steps": float(
                observer_lora.get("admit_relaxed_steps", 0) or 0
            ),
            "admit_tightened_steps": float(
                observer_lora.get("admit_tightened_steps", 0) or 0
            ),
            "prefill_relaxed_steps": float(
                observer_lora.get("prefill_relaxed_steps", 0) or 0
            ),
            "prefill_tightened_steps": float(
                observer_lora.get("prefill_tightened_steps", 0) or 0
            ),
            "decode_relaxed_steps": float(
                observer_lora.get("decode_relaxed_steps", 0) or 0
            ),
            "decode_tightened_steps": float(
                observer_lora.get("decode_tightened_steps", 0) or 0
            ),
            "prefill_step_count": lora_prefill_steps,
            "decode_step_count": lora_decode_steps,
            "mixed_lora_prefill_step_rate": (
                float(observer_lora.get("mixed_lora_prefill_steps", 0) or 0)
                / step_count
                if step_count
                else 0.0
            ),
            "mixed_lora_decode_step_rate": (
                float(observer_lora.get("mixed_lora_decode_steps", 0) or 0) / step_count
                if step_count
                else 0.0
            ),
            "prefill_adapter_count": float(len(lora_prefill_adapters)),
            "decode_adapter_count": float(len(lora_decode_adapters)),
            "backlog_adapter_count": float(len(lora_backlog_adapters)),
            "prefill_locality_rate": (
                1.0
                - (
                    float(observer_lora.get("mixed_lora_prefill_steps", 0) or 0)
                    / lora_prefill_steps
                )
                if lora_prefill_steps
                else 0.0
            ),
            "decode_locality_rate": (
                1.0
                - (
                    float(observer_lora.get("mixed_lora_decode_steps", 0) or 0)
                    / lora_decode_steps
                )
                if lora_decode_steps
                else 0.0
            ),
            "admitted_adapter_share": admitted_adapter_share,
            "backlog_adapter_share": backlog_adapter_share,
            "adapter_fairness_gap": adapter_fairness_gap,
            "max_adapter_fairness_gap": (
                max(
                    (abs(value) for value in adapter_fairness_gap.values()), default=0.0
                )
            ),
            "mean_abs_adapter_fairness_gap": (
                sum(abs(value) for value in adapter_fairness_gap.values())
                / max(1, len(adapter_fairness_gap))
            ),
            "admit_relaxed_rate": (
                float(observer_lora.get("admit_relaxed_steps", 0) or 0) / step_count
                if step_count
                else 0.0
            ),
            "admit_tightened_rate": (
                float(observer_lora.get("admit_tightened_steps", 0) or 0) / step_count
                if step_count
                else 0.0
            ),
            "prefill_relaxed_rate": (
                float(observer_lora.get("prefill_relaxed_steps", 0) or 0) / step_count
                if step_count
                else 0.0
            ),
            "prefill_tightened_rate": (
                float(observer_lora.get("prefill_tightened_steps", 0) or 0) / step_count
                if step_count
                else 0.0
            ),
            "decode_relaxed_rate": (
                float(observer_lora.get("decode_relaxed_steps", 0) or 0) / step_count
                if step_count
                else 0.0
            ),
            "decode_tightened_rate": (
                float(observer_lora.get("decode_tightened_steps", 0) or 0) / step_count
                if step_count
                else 0.0
            ),
        },
        "multimodal": {
            "request_count": float(observer_multimodal.get("requests", 0) or 0),
            "image_count": float(observer_multimodal.get("images", 0) or 0),
            "multimodal_lora_request_count": float(
                observer_multimodal.get("multimodal_lora_requests", 0) or 0
            ),
            "queued_request_count": float(
                observer_multimodal.get("queued_requests", 0) or 0
            ),
            "admitted_request_count": float(
                observer_multimodal.get("admitted_requests", 0) or 0
            ),
            "prefill_request_count": float(
                observer_multimodal.get("prefill_requests", 0) or 0
            ),
            "decode_request_count": float(
                observer_multimodal.get("decode_requests", 0) or 0
            ),
            "p95_queue_wait_s": float(
                observer_multimodal.get("p95_queue_wait_s", 0.0) or 0.0
            ),
            "multimodal_lora_prefill_count": float(
                observer_multimodal.get("prefill_multimodal_lora_requests", 0) or 0
            ),
            "mixed_multimodal_lora_prefill_steps": float(
                observer_multimodal.get("mixed_multimodal_lora_prefill_steps", 0) or 0
            ),
            "avg_effective_prefill_multimodal_limit": float(
                observer_multimodal.get("avg_effective_prefill_multimodal_limit", 0.0)
                or 0.0
            ),
            "prefill_multimodal_limit_relaxed_steps": float(
                observer_multimodal.get("prefill_multimodal_limit_relaxed_steps", 0)
                or 0
            ),
            "prefill_multimodal_limit_tightened_steps": float(
                observer_multimodal.get("prefill_multimodal_limit_tightened_steps", 0)
                or 0
            ),
            "prefill_multimodal_limit_triggered_steps": float(
                observer_multimodal.get("prefill_multimodal_limit_triggered_steps", 0)
                or 0
            ),
            "prefix_cache_hits": float(
                observer_multimodal.get("prefix_cache_hits", 0) or 0
            ),
            "prefix_cache_misses": float(
                observer_multimodal.get("prefix_cache_misses", 0) or 0
            ),
            "prefix_cache_saved_prefill_tokens": float(
                observer_multimodal.get("prefix_cache_saved_prefill_tokens", 0) or 0
            ),
            "admit_multimodal_lora_limit_triggered_steps": float(
                observer_multimodal.get(
                    "admit_multimodal_lora_limit_triggered_steps", 0
                )
                or 0
            ),
            "prefill_multimodal_lora_limit_triggered_steps": float(
                observer_multimodal.get(
                    "prefill_multimodal_lora_limit_triggered_steps", 0
                )
                or 0
            ),
            "prefill_multimodal_lora_limit_relaxed_steps": float(
                observer_multimodal.get(
                    "prefill_multimodal_lora_limit_relaxed_steps", 0
                )
                or 0
            ),
            "prefill_multimodal_lora_limit_relaxed_by_fairness_steps": float(
                observer_multimodal.get(
                    "prefill_multimodal_lora_limit_relaxed_by_fairness_steps", 0
                )
                or 0
            ),
            "prefill_multimodal_lora_limit_tightened_steps": float(
                observer_multimodal.get(
                    "prefill_multimodal_lora_limit_tightened_steps", 0
                )
                or 0
            ),
            "prefill_multimodal_lora_limit_tightened_by_locality_steps": float(
                observer_multimodal.get(
                    "prefill_multimodal_lora_limit_tightened_by_locality_steps", 0
                )
                or 0
            ),
            "preempted_prefill_requests": preempted_multimodal_prefills,
            "prepared_request_count": float(
                backend_multimodal.get("prepared_requests", 0) or 0
            ),
            "prepared_image_count": float(
                backend_multimodal.get("prepared_images", 0) or 0
            ),
            "prepare_failure_count": float(
                backend_multimodal.get("prepare_failures", 0) or 0
            ),
            "embedding_request_count": float(
                backend_multimodal.get("embedding_requests", 0) or 0
            ),
            "embeddings_computed": float(
                backend_multimodal.get("embeddings_computed", 0) or 0
            ),
            "avg_embedding_feature_dim": float(
                backend_multimodal.get("avg_embedding_feature_dim", 0.0) or 0.0
            ),
            "images_per_request": (
                float(observer_multimodal.get("images", 0) or 0)
                / float(observer_multimodal.get("requests", 0) or 1)
                if float(observer_multimodal.get("requests", 0) or 0) > 0
                else 0.0
            ),
            "prepared_images_per_request": (
                float(backend_multimodal.get("prepared_images", 0) or 0)
                / float(backend_multimodal.get("prepared_requests", 0) or 1)
                if float(backend_multimodal.get("prepared_requests", 0) or 0) > 0
                else 0.0
            ),
            "embedding_compute_rate": (
                float(backend_multimodal.get("embeddings_computed", 0) or 0)
                / float(backend_multimodal.get("embedding_requests", 0) or 1)
                if float(backend_multimodal.get("embedding_requests", 0) or 0) > 0
                else 0.0
            ),
            "preempted_prefill_rate": (
                preempted_multimodal_prefills
                / float(observer_multimodal.get("prefill_requests", 0) or 1)
                if float(observer_multimodal.get("prefill_requests", 0) or 0) > 0
                else 0.0
            ),
            "multimodal_lora_request_rate": (
                float(observer_multimodal.get("multimodal_lora_requests", 0) or 0)
                / float(observer_multimodal.get("requests", 0) or 1)
                if float(observer_multimodal.get("requests", 0) or 0) > 0
                else 0.0
            ),
            "multimodal_lora_prefill_rate": (
                float(
                    observer_multimodal.get("prefill_multimodal_lora_requests", 0) or 0
                )
                / float(observer_multimodal.get("prefill_requests", 0) or 1)
                if float(observer_multimodal.get("prefill_requests", 0) or 0) > 0
                else 0.0
            ),
            "prefix_cache_hit_rate": float(
                observer_multimodal.get("prefix_cache_hit_rate", 0.0) or 0.0
            ),
            "saved_prefill_tokens_per_request": (
                float(
                    observer_multimodal.get("prefix_cache_saved_prefill_tokens", 0) or 0
                )
                / float(observer_multimodal.get("requests", 0) or 1)
                if float(observer_multimodal.get("requests", 0) or 0) > 0
                else 0.0
            ),
            "prefill_multimodal_limit_relaxed_rate": (
                float(
                    observer_multimodal.get("prefill_multimodal_limit_relaxed_steps", 0)
                    or 0
                )
                / step_count
                if step_count
                else 0.0
            ),
            "prefill_multimodal_limit_tightened_rate": (
                float(
                    observer_multimodal.get(
                        "prefill_multimodal_limit_tightened_steps", 0
                    )
                    or 0
                )
                / step_count
                if step_count
                else 0.0
            ),
            "prefill_multimodal_limit_triggered_rate": (
                float(
                    observer_multimodal.get(
                        "prefill_multimodal_limit_triggered_steps", 0
                    )
                    or 0
                )
                / step_count
                if step_count
                else 0.0
            ),
            "mixed_multimodal_lora_batch_ratio": float(
                observer_multimodal.get("mixed_multimodal_lora_prefill_ratio", 0.0)
                or 0.0
            ),
            "avg_effective_admit_multimodal_lora_limit": float(
                observer_multimodal.get(
                    "avg_effective_admit_multimodal_lora_limit", 0.0
                )
                or 0.0
            ),
            "avg_effective_prefill_multimodal_lora_limit": float(
                observer_multimodal.get(
                    "avg_effective_prefill_multimodal_lora_limit", 0.0
                )
                or 0.0
            ),
            "avg_effective_decode_multimodal_lora_limit": float(
                observer_multimodal.get(
                    "avg_effective_decode_multimodal_lora_limit", 0.0
                )
                or 0.0
            ),
            "avg_prefill_multimodal_lora_max_fairness_gap": float(
                observer_multimodal.get(
                    "avg_prefill_multimodal_lora_max_fairness_gap", 0.0
                )
                or 0.0
            ),
            "avg_decode_multimodal_lora_max_fairness_gap": float(
                observer_multimodal.get(
                    "avg_decode_multimodal_lora_max_fairness_gap", 0.0
                )
                or 0.0
            ),
            "admit_multimodal_lora_limit_triggered_rate": (
                float(
                    observer_multimodal.get(
                        "admit_multimodal_lora_limit_triggered_steps", 0
                    )
                    or 0
                )
                / step_count
                if step_count
                else 0.0
            ),
            "prefill_multimodal_lora_limit_triggered_rate": (
                float(
                    observer_multimodal.get(
                        "prefill_multimodal_lora_limit_triggered_steps", 0
                    )
                    or 0
                )
                / step_count
                if step_count
                else 0.0
            ),
            "prefill_multimodal_lora_limit_relaxed_rate": (
                float(
                    observer_multimodal.get(
                        "prefill_multimodal_lora_limit_relaxed_steps", 0
                    )
                    or 0
                )
                / step_count
                if step_count
                else 0.0
            ),
            "prefill_multimodal_lora_limit_relaxed_by_fairness_rate": (
                float(
                    observer_multimodal.get(
                        "prefill_multimodal_lora_limit_relaxed_by_fairness_steps", 0
                    )
                    or 0
                )
                / step_count
                if step_count
                else 0.0
            ),
            "prefill_multimodal_lora_limit_tightened_rate": (
                float(
                    observer_multimodal.get(
                        "prefill_multimodal_lora_limit_tightened_steps", 0
                    )
                    or 0
                )
                / step_count
                if step_count
                else 0.0
            ),
            "prefill_multimodal_lora_limit_tightened_by_locality_rate": (
                float(
                    observer_multimodal.get(
                        "prefill_multimodal_lora_limit_tightened_by_locality_steps",
                        0,
                    )
                    or 0
                )
                / step_count
                if step_count
                else 0.0
            ),
            "decode_multimodal_lora_limit_triggered_steps": float(
                observer_multimodal.get(
                    "decode_multimodal_lora_limit_triggered_steps", 0
                )
                or 0
            ),
            "decode_multimodal_lora_limit_relaxed_steps": float(
                observer_multimodal.get("decode_multimodal_lora_limit_relaxed_steps", 0)
                or 0
            ),
            "decode_multimodal_lora_limit_relaxed_by_fairness_steps": float(
                observer_multimodal.get(
                    "decode_multimodal_lora_limit_relaxed_by_fairness_steps", 0
                )
                or 0
            ),
            "decode_multimodal_lora_limit_tightened_steps": float(
                observer_multimodal.get(
                    "decode_multimodal_lora_limit_tightened_steps", 0
                )
                or 0
            ),
            "decode_multimodal_lora_limit_tightened_by_locality_steps": float(
                observer_multimodal.get(
                    "decode_multimodal_lora_limit_tightened_by_locality_steps", 0
                )
                or 0
            ),
            "decode_multimodal_lora_limit_triggered_rate": (
                float(
                    observer_multimodal.get(
                        "decode_multimodal_lora_limit_triggered_steps", 0
                    )
                    or 0
                )
                / step_count
                if step_count
                else 0.0
            ),
            "decode_multimodal_lora_limit_relaxed_rate": (
                float(
                    observer_multimodal.get(
                        "decode_multimodal_lora_limit_relaxed_steps", 0
                    )
                    or 0
                )
                / step_count
                if step_count
                else 0.0
            ),
            "decode_multimodal_lora_limit_relaxed_by_fairness_rate": (
                float(
                    observer_multimodal.get(
                        "decode_multimodal_lora_limit_relaxed_by_fairness_steps", 0
                    )
                    or 0
                )
                / step_count
                if step_count
                else 0.0
            ),
            "decode_multimodal_lora_limit_tightened_rate": (
                float(
                    observer_multimodal.get(
                        "decode_multimodal_lora_limit_tightened_steps", 0
                    )
                    or 0
                )
                / step_count
                if step_count
                else 0.0
            ),
            "decode_multimodal_lora_limit_tightened_by_locality_rate": (
                float(
                    observer_multimodal.get(
                        "decode_multimodal_lora_limit_tightened_by_locality_steps", 0
                    )
                    or 0
                )
                / step_count
                if step_count
                else 0.0
            ),
        },
    }


def _normalized_share_map(values: object) -> dict[str, float]:
    if not isinstance(values, dict):
        return {}
    normalized = {str(key): float(value or 0.0) for key, value in values.items()}
    total = sum(normalized.values())
    if total <= 0:
        return {key: 0.0 for key in normalized}
    return {key: value / total for key, value in normalized.items()}


def _share_gap_map(
    target_share: dict[str, float],
    baseline_share: dict[str, float],
) -> dict[str, float]:
    keys = sorted(set(target_share) | set(baseline_share))
    return {
        key: float(target_share.get(key, 0.0) or 0.0)
        - float(baseline_share.get(key, 0.0) or 0.0)
        for key in keys
    }


def _derive_runtime_phase_diffs(
    phases: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    warmup = phases.get("warmup", {})
    benchmark = phases.get("benchmark", {})
    if not isinstance(warmup, dict) or not isinstance(benchmark, dict):
        return {}

    warmup_metrics = warmup.get("derived_metrics", {})
    benchmark_metrics = benchmark.get("derived_metrics", {})
    if not isinstance(warmup_metrics, dict) or not isinstance(benchmark_metrics, dict):
        return {}

    warmup_async = warmup_metrics.get("async_driver", {})
    benchmark_async = benchmark_metrics.get("async_driver", {})
    warmup_prefix = warmup_metrics.get("prefix_cache", {})
    benchmark_prefix = benchmark_metrics.get("prefix_cache", {})
    warmup_preemption = warmup_metrics.get("preemption", {})
    benchmark_preemption = benchmark_metrics.get("preemption", {})
    warmup_fairness = warmup_metrics.get("fairness", {})
    benchmark_fairness = benchmark_metrics.get("fairness", {})
    warmup_lora = warmup_metrics.get("lora", {})
    benchmark_lora = benchmark_metrics.get("lora", {})
    warmup_multimodal = warmup_metrics.get("multimodal", {})
    benchmark_multimodal = benchmark_metrics.get("multimodal", {})
    if not isinstance(warmup_async, dict) or not isinstance(benchmark_async, dict):
        warmup_async, benchmark_async = {}, {}
    if not isinstance(warmup_prefix, dict) or not isinstance(benchmark_prefix, dict):
        warmup_prefix, benchmark_prefix = {}, {}
    if not isinstance(warmup_preemption, dict) or not isinstance(
        benchmark_preemption, dict
    ):
        warmup_preemption, benchmark_preemption = {}, {}
    if not isinstance(warmup_fairness, dict) or not isinstance(
        benchmark_fairness, dict
    ):
        warmup_fairness, benchmark_fairness = {}, {}
    if not isinstance(warmup_lora, dict) or not isinstance(benchmark_lora, dict):
        warmup_lora, benchmark_lora = {}, {}
    if not isinstance(warmup_multimodal, dict) or not isinstance(
        benchmark_multimodal, dict
    ):
        warmup_multimodal, benchmark_multimodal = {}, {}

    async_keys = (
        "steps",
        "backpressure_sleeps",
        "idle_waits",
        "background_errors",
        "backpressure_sleep_rate",
        "idle_wait_rate",
        "background_error_rate",
        "observer_step_gap",
    )
    async_delta = {}
    for key in async_keys:
        async_delta[key] = float(benchmark_async.get(key, 0.0) or 0.0) - float(
            warmup_async.get(key, 0.0) or 0.0
        )

    prefix_keys = (
        "request_count",
        "lookup_hit_rate",
        "materialized_hit_rate",
        "saved_prefill_tokens_per_request",
        "saved_prefill_tokens_per_materialized_hit",
        "lookup_cost_per_request",
        "lookup_candidates_per_request",
    )
    prefix_delta = {}
    for key in prefix_keys:
        prefix_delta[key] = float(benchmark_prefix.get(key, 0.0) or 0.0) - float(
            warmup_prefix.get(key, 0.0) or 0.0
        )

    preemption_keys = (
        "step_count",
        "preempted_steps",
        "preempted_prefill_requests",
        "preempted_multimodal_prefill_requests",
        "protected_multimodal_prefix_steps",
        "protected_multimodal_prefix_prefill_requests",
        "preempted_step_rate",
        "preempted_prefill_requests_per_step",
        "preempted_multimodal_prefill_requests_per_step",
        "protected_multimodal_prefix_step_rate",
        "protected_multimodal_prefix_prefill_requests_per_step",
    )
    preemption_delta = {}
    for key in preemption_keys:
        preemption_delta[key] = float(
            benchmark_preemption.get(key, 0.0) or 0.0
        ) - float(warmup_preemption.get(key, 0.0) or 0.0)

    fairness_scalar_keys = (
        "step_count",
        "admitted_requests",
        "starvation_protected_steps",
        "fairness_guardrail_triggered_steps",
        "starvation_protected_step_rate",
        "fairness_guardrail_triggered_step_rate",
        "avg_admitted_queue_wait_s",
        "p95_queue_wait_s",
        "max_queue_wait_s",
    )
    fairness_delta = {}
    for key in fairness_scalar_keys:
        fairness_delta[key] = float(benchmark_fairness.get(key, 0.0) or 0.0) - float(
            warmup_fairness.get(key, 0.0) or 0.0
        )
    warmup_fairness_p95 = warmup_fairness.get("per_class_p95_queue_wait_s", {})
    benchmark_fairness_p95 = benchmark_fairness.get("per_class_p95_queue_wait_s", {})
    if not isinstance(warmup_fairness_p95, dict):
        warmup_fairness_p95 = {}
    if not isinstance(benchmark_fairness_p95, dict):
        benchmark_fairness_p95 = {}
    per_class_keys = sorted(set(warmup_fairness_p95) | set(benchmark_fairness_p95))
    fairness_delta["per_class_p95_queue_wait_s"] = {
        str(key): float(benchmark_fairness_p95.get(key, 0.0) or 0.0)
        - float(warmup_fairness_p95.get(key, 0.0) or 0.0)
        for key in per_class_keys
    }

    lora_keys = (
        "registered_adapters",
        "active_adapters",
        "active_requests",
        "total_routed_requests",
        "adapter_count_observed",
        "mixed_lora_prefill_steps",
        "mixed_lora_decode_steps",
        "admit_relaxed_steps",
        "admit_tightened_steps",
        "prefill_relaxed_steps",
        "prefill_tightened_steps",
        "decode_relaxed_steps",
        "decode_tightened_steps",
        "prefill_step_count",
        "decode_step_count",
        "mixed_lora_prefill_step_rate",
        "mixed_lora_decode_step_rate",
        "prefill_adapter_count",
        "decode_adapter_count",
        "backlog_adapter_count",
        "prefill_locality_rate",
        "decode_locality_rate",
        "max_adapter_fairness_gap",
        "mean_abs_adapter_fairness_gap",
        "admit_relaxed_rate",
        "admit_tightened_rate",
        "prefill_relaxed_rate",
        "prefill_tightened_rate",
        "decode_relaxed_rate",
        "decode_tightened_rate",
    )
    lora_delta = {}
    for key in lora_keys:
        lora_delta[key] = float(benchmark_lora.get(key, 0.0) or 0.0) - float(
            warmup_lora.get(key, 0.0) or 0.0
        )
    warmup_adapter_fairness = warmup_lora.get("adapter_fairness_gap", {})
    benchmark_adapter_fairness = benchmark_lora.get("adapter_fairness_gap", {})
    if not isinstance(warmup_adapter_fairness, dict):
        warmup_adapter_fairness = {}
    if not isinstance(benchmark_adapter_fairness, dict):
        benchmark_adapter_fairness = {}
    fairness_keys = sorted(
        set(warmup_adapter_fairness) | set(benchmark_adapter_fairness)
    )
    lora_delta["adapter_fairness_gap"] = {
        str(key): float(benchmark_adapter_fairness.get(key, 0.0) or 0.0)
        - float(warmup_adapter_fairness.get(key, 0.0) or 0.0)
        for key in fairness_keys
    }

    multimodal_keys = (
        "request_count",
        "multimodal_lora_request_count",
        "image_count",
        "queued_request_count",
        "admitted_request_count",
        "prefill_request_count",
        "multimodal_lora_prefill_count",
        "decode_request_count",
        "p95_queue_wait_s",
        "mixed_multimodal_lora_prefill_steps",
        "avg_effective_prefill_multimodal_limit",
        "prefill_multimodal_limit_relaxed_steps",
        "prefill_multimodal_limit_tightened_steps",
        "prefill_multimodal_limit_triggered_steps",
        "prefix_cache_hits",
        "prefix_cache_misses",
        "prefix_cache_saved_prefill_tokens",
        "admit_multimodal_lora_limit_triggered_steps",
        "prefill_multimodal_lora_limit_triggered_steps",
        "preempted_prefill_requests",
        "prepared_request_count",
        "prepared_image_count",
        "prepare_failure_count",
        "embedding_request_count",
        "embeddings_computed",
        "avg_embedding_feature_dim",
        "images_per_request",
        "prepared_images_per_request",
        "embedding_compute_rate",
        "preempted_prefill_rate",
        "multimodal_lora_request_rate",
        "multimodal_lora_prefill_rate",
        "prefix_cache_hit_rate",
        "saved_prefill_tokens_per_request",
        "prefill_multimodal_limit_relaxed_rate",
        "prefill_multimodal_limit_tightened_rate",
        "prefill_multimodal_limit_triggered_rate",
        "mixed_multimodal_lora_batch_ratio",
        "avg_effective_admit_multimodal_lora_limit",
        "avg_effective_prefill_multimodal_lora_limit",
        "avg_effective_decode_multimodal_lora_limit",
        "avg_prefill_multimodal_lora_max_fairness_gap",
        "avg_decode_multimodal_lora_max_fairness_gap",
        "admit_multimodal_lora_limit_triggered_rate",
        "prefill_multimodal_lora_limit_triggered_rate",
        "prefill_multimodal_lora_limit_relaxed_steps",
        "prefill_multimodal_lora_limit_tightened_steps",
        "prefill_multimodal_lora_limit_relaxed_rate",
        "prefill_multimodal_lora_limit_tightened_rate",
        "prefill_multimodal_lora_limit_relaxed_by_fairness_steps",
        "prefill_multimodal_lora_limit_tightened_by_locality_steps",
        "prefill_multimodal_lora_limit_relaxed_by_fairness_rate",
        "prefill_multimodal_lora_limit_tightened_by_locality_rate",
        "decode_multimodal_lora_limit_triggered_steps",
        "decode_multimodal_lora_limit_relaxed_steps",
        "decode_multimodal_lora_limit_tightened_steps",
        "decode_multimodal_lora_limit_relaxed_by_fairness_steps",
        "decode_multimodal_lora_limit_tightened_by_locality_steps",
        "decode_multimodal_lora_limit_triggered_rate",
        "decode_multimodal_lora_limit_relaxed_rate",
        "decode_multimodal_lora_limit_tightened_rate",
        "decode_multimodal_lora_limit_relaxed_by_fairness_rate",
        "decode_multimodal_lora_limit_tightened_by_locality_rate",
    )
    multimodal_delta = {}
    for key in multimodal_keys:
        multimodal_delta[key] = float(
            benchmark_multimodal.get(key, 0.0) or 0.0
        ) - float(warmup_multimodal.get(key, 0.0) or 0.0)

    return {
        "baseline_phase": "warmup",
        "target_phase": "benchmark",
        "async_driver": {
            "warmup": dict(warmup_async),
            "benchmark": dict(benchmark_async),
            "benchmark_delta": async_delta,
        },
        "prefix_cache": {
            "warmup": dict(warmup_prefix),
            "benchmark": dict(benchmark_prefix),
            "benchmark_delta": prefix_delta,
        },
        "preemption": {
            "warmup": dict(warmup_preemption),
            "benchmark": dict(benchmark_preemption),
            "benchmark_delta": preemption_delta,
        },
        "fairness": {
            "warmup": dict(warmup_fairness),
            "benchmark": dict(benchmark_fairness),
            "benchmark_delta": fairness_delta,
        },
        "lora": {
            "warmup": dict(warmup_lora),
            "benchmark": dict(benchmark_lora),
            "benchmark_delta": lora_delta,
        },
        "multimodal": {
            "warmup": dict(warmup_multimodal),
            "benchmark": dict(benchmark_multimodal),
            "benchmark_delta": multimodal_delta,
        },
    }


def _format_runtime_phase_diff_summary(
    phase_diffs: Dict[str, object],
) -> list[str]:
    if not isinstance(phase_diffs, dict):
        return []

    lines: list[str] = []
    async_driver = phase_diffs.get("async_driver", {})
    if isinstance(async_driver, dict):
        async_delta = async_driver.get("benchmark_delta", {})
        if isinstance(async_delta, dict):
            lines.append(
                "  RUNTIME(async): "
                f"steps_delta={float(async_delta.get('steps', 0.0) or 0.0):+.0f}, "
                f"backpressure_delta={float(async_delta.get('backpressure_sleeps', 0.0) or 0.0):+.0f}, "
                f"idle_wait_delta={float(async_delta.get('idle_waits', 0.0) or 0.0):+.0f}, "
                f"error_delta={float(async_delta.get('background_errors', 0.0) or 0.0):+.0f}, "
                f"sleep_rate_delta={float(async_delta.get('backpressure_sleep_rate', 0.0) or 0.0):+.3f}"
            )

    prefix_cache = phase_diffs.get("prefix_cache", {})
    if isinstance(prefix_cache, dict):
        prefix_delta = prefix_cache.get("benchmark_delta", {})
        if isinstance(prefix_delta, dict):
            lines.append(
                "  RUNTIME(prefix): "
                f"mat_hit_rate_delta={float(prefix_delta.get('materialized_hit_rate', 0.0) or 0.0):+.3f}, "
                f"saved_prefill_tok_per_req_delta={float(prefix_delta.get('saved_prefill_tokens_per_request', 0.0) or 0.0):+.3f}, "
                f"lookup_cost_per_req_delta={float(prefix_delta.get('lookup_cost_per_request', 0.0) or 0.0):+.3f}"
            )

    preemption = phase_diffs.get("preemption", {})
    if isinstance(preemption, dict):
        preemption_delta = preemption.get("benchmark_delta", {})
        if isinstance(preemption_delta, dict):
            lines.append(
                "  RUNTIME(preempt): "
                f"step_rate_delta={float(preemption_delta.get('preempted_step_rate', 0.0) or 0.0):+.3f}, "
                f"prefills_per_step_delta={float(preemption_delta.get('preempted_prefill_requests_per_step', 0.0) or 0.0):+.3f}, "
                f"mm_prefills_per_step_delta={float(preemption_delta.get('preempted_multimodal_prefill_requests_per_step', 0.0) or 0.0):+.3f}, "
                f"mm_prefix_protect_rate_delta={float(preemption_delta.get('protected_multimodal_prefix_step_rate', 0.0) or 0.0):+.3f}"
            )

    fairness = phase_diffs.get("fairness", {})
    if isinstance(fairness, dict):
        fairness_delta = fairness.get("benchmark_delta", {})
        if isinstance(fairness_delta, dict):
            lines.append(
                "  RUNTIME(fair): "
                f"guardrail_rate_delta={float(fairness_delta.get('fairness_guardrail_triggered_step_rate', 0.0) or 0.0):+.3f}, "
                f"starvation_rate_delta={float(fairness_delta.get('starvation_protected_step_rate', 0.0) or 0.0):+.3f}, "
                f"p95_queue_wait_delta={float(fairness_delta.get('p95_queue_wait_s', 0.0) or 0.0):+.3f}s"
            )
            per_class = fairness_delta.get("per_class_p95_queue_wait_s", {})
            if isinstance(per_class, dict) and per_class:
                formatted = ", ".join(
                    f"{key}={float(value or 0.0):+.3f}s"
                    for key, value in sorted(per_class.items())
                )
                lines.append(f"  RUNTIME(fair,p95_by_class): {formatted}")

    lora = phase_diffs.get("lora", {})
    if isinstance(lora, dict):
        lora_delta = lora.get("benchmark_delta", {})
        if isinstance(lora_delta, dict):
            lines.append(
                "  RUNTIME(lora): "
                f"mixed_prefill_rate_delta={float(lora_delta.get('mixed_lora_prefill_step_rate', 0.0) or 0.0):+.3f}, "
                f"mixed_decode_rate_delta={float(lora_delta.get('mixed_lora_decode_step_rate', 0.0) or 0.0):+.3f}, "
                f"routed_req_delta={float(lora_delta.get('total_routed_requests', 0.0) or 0.0):+.3f}"
            )
            lines.append(
                "  RUNTIME(lora,adapters): "
                f"prefill_delta={float(lora_delta.get('prefill_adapter_count', 0.0) or 0.0):+.3f}, "
                f"decode_delta={float(lora_delta.get('decode_adapter_count', 0.0) or 0.0):+.3f}, "
                f"backlog_delta={float(lora_delta.get('backlog_adapter_count', 0.0) or 0.0):+.3f}"
            )
            lines.append(
                "  RUNTIME(lora,fair): "
                f"prefill_locality_delta={float(lora_delta.get('prefill_locality_rate', 0.0) or 0.0):+.3f}, "
                f"decode_locality_delta={float(lora_delta.get('decode_locality_rate', 0.0) or 0.0):+.3f}, "
                f"max_fair_gap_delta={float(lora_delta.get('max_adapter_fairness_gap', 0.0) or 0.0):+.3f}"
            )
            lines.append(
                "  RUNTIME(lora,adaptive): "
                f"admit_relaxed_rate_delta={float(lora_delta.get('admit_relaxed_rate', 0.0) or 0.0):+.3f}, "
                f"prefill_tightened_rate_delta={float(lora_delta.get('prefill_tightened_rate', 0.0) or 0.0):+.3f}, "
                f"decode_relaxed_rate_delta={float(lora_delta.get('decode_relaxed_rate', 0.0) or 0.0):+.3f}"
            )
            per_adapter = lora_delta.get("adapter_fairness_gap", {})
            if isinstance(per_adapter, dict) and per_adapter:
                formatted = ", ".join(
                    f"{key}={float(value or 0.0):+.3f}"
                    for key, value in sorted(per_adapter.items())
                )
                lines.append(f"  RUNTIME(lora,fair_by_adapter): {formatted}")

    multimodal = phase_diffs.get("multimodal", {})
    if isinstance(multimodal, dict):
        multimodal_delta = multimodal.get("benchmark_delta", {})
        if isinstance(multimodal_delta, dict):
            lines.append(
                "  RUNTIME(multimodal): "
                f"req_delta={float(multimodal_delta.get('request_count', 0.0) or 0.0):+.3f}, "
                f"mm_lora_req_delta={float(multimodal_delta.get('multimodal_lora_request_count', 0.0) or 0.0):+.3f}, "
                f"mm_lora_prefill_delta={float(multimodal_delta.get('multimodal_lora_prefill_count', 0.0) or 0.0):+.3f}, "
                f"queued_delta={float(multimodal_delta.get('queued_request_count', 0.0) or 0.0):+.3f}, "
                f"p95_wait_delta={float(multimodal_delta.get('p95_queue_wait_s', 0.0) or 0.0):+.3f}s, "
                f"img_per_req_delta={float(multimodal_delta.get('images_per_request', 0.0) or 0.0):+.3f}, "
                f"embed_rate_delta={float(multimodal_delta.get('embedding_compute_rate', 0.0) or 0.0):+.3f}, "
                f"prefix_hit_rate_delta={float(multimodal_delta.get('prefix_cache_hit_rate', 0.0) or 0.0):+.3f}, "
                f"saved_prefill_tok_per_req_delta={float(multimodal_delta.get('saved_prefill_tokens_per_request', 0.0) or 0.0):+.3f}, "
                f"mm_prefill_limit_delta={float(multimodal_delta.get('avg_effective_prefill_multimodal_limit', 0.0) or 0.0):+.3f}, "
                f"mm_prefill_relaxed_rate_delta={float(multimodal_delta.get('prefill_multimodal_limit_relaxed_rate', 0.0) or 0.0):+.3f}, "
                f"mm_prefill_tightened_rate_delta={float(multimodal_delta.get('prefill_multimodal_limit_tightened_rate', 0.0) or 0.0):+.3f}, "
                f"mm_preempt_rate_delta={float(multimodal_delta.get('preempted_prefill_rate', 0.0) or 0.0):+.3f}, "
                f"mm_lora_batch_ratio_delta={float(multimodal_delta.get('mixed_multimodal_lora_batch_ratio', 0.0) or 0.0):+.3f}, "
                f"mm_lora_admit_limit_delta={float(multimodal_delta.get('avg_effective_admit_multimodal_lora_limit', 0.0) or 0.0):+.3f}, "
                f"mm_lora_prefill_limit_delta={float(multimodal_delta.get('avg_effective_prefill_multimodal_lora_limit', 0.0) or 0.0):+.3f}, "
                f"mm_lora_decode_limit_delta={float(multimodal_delta.get('avg_effective_decode_multimodal_lora_limit', 0.0) or 0.0):+.3f}, "
                f"mm_lora_gap_delta={float(multimodal_delta.get('avg_prefill_multimodal_lora_max_fairness_gap', 0.0) or 0.0):+.3f}, "
                f"mm_lora_decode_gap_delta={float(multimodal_delta.get('avg_decode_multimodal_lora_max_fairness_gap', 0.0) or 0.0):+.3f}, "
                f"mm_lora_admit_trigger_rate_delta={float(multimodal_delta.get('admit_multimodal_lora_limit_triggered_rate', 0.0) or 0.0):+.3f}, "
                f"mm_lora_prefill_trigger_rate_delta={float(multimodal_delta.get('prefill_multimodal_lora_limit_triggered_rate', 0.0) or 0.0):+.3f}, "
                f"mm_lora_decode_trigger_rate_delta={float(multimodal_delta.get('decode_multimodal_lora_limit_triggered_rate', 0.0) or 0.0):+.3f}, "
                f"mm_lora_prefill_relaxed_rate_delta={float(multimodal_delta.get('prefill_multimodal_lora_limit_relaxed_rate', 0.0) or 0.0):+.3f}, "
                f"mm_lora_prefill_tightened_rate_delta={float(multimodal_delta.get('prefill_multimodal_lora_limit_tightened_rate', 0.0) or 0.0):+.3f}, "
                f"mm_lora_fair_relax_rate_delta={float(multimodal_delta.get('prefill_multimodal_lora_limit_relaxed_by_fairness_rate', 0.0) or 0.0):+.3f}, "
                f"mm_lora_local_tighten_rate_delta={float(multimodal_delta.get('prefill_multimodal_lora_limit_tightened_by_locality_rate', 0.0) or 0.0):+.3f}, "
                f"mm_lora_decode_relaxed_rate_delta={float(multimodal_delta.get('decode_multimodal_lora_limit_relaxed_rate', 0.0) or 0.0):+.3f}, "
                f"mm_lora_decode_tightened_rate_delta={float(multimodal_delta.get('decode_multimodal_lora_limit_tightened_rate', 0.0) or 0.0):+.3f}, "
                f"mm_lora_decode_fair_relax_rate_delta={float(multimodal_delta.get('decode_multimodal_lora_limit_relaxed_by_fairness_rate', 0.0) or 0.0):+.3f}, "
                f"mm_lora_decode_local_tighten_rate_delta={float(multimodal_delta.get('decode_multimodal_lora_limit_tightened_by_locality_rate', 0.0) or 0.0):+.3f}"
            )

    return lines


def _format_runtime_snapshot_summary(snapshot: Dict[str, object]) -> list[str]:
    if not isinstance(snapshot, dict):
        return []
    derived = snapshot.get("derived_metrics", {})
    if not isinstance(derived, dict):
        return []
    lines: list[str] = []
    async_driver = derived.get("async_driver", {})
    lora = derived.get("lora", {})
    multimodal = derived.get("multimodal", {})
    if isinstance(async_driver, dict):
        lines.append(
            "  RUNTIME(async,current): "
            f"steps={float(async_driver.get('steps', 0.0) or 0.0):.0f}, "
            f"backpressure_sleeps={float(async_driver.get('backpressure_sleeps', 0.0) or 0.0):.0f}, "
            f"idle_waits={float(async_driver.get('idle_waits', 0.0) or 0.0):.0f}, "
            f"errors={float(async_driver.get('background_errors', 0.0) or 0.0):.0f}, "
            f"sleep_rate={float(async_driver.get('backpressure_sleep_rate', 0.0) or 0.0):.3f}"
        )
    if isinstance(lora, dict):
        lines.append(
            "  RUNTIME(lora,current): "
            f"prefill_locality={float(lora.get('prefill_locality_rate', 0.0) or 0.0):.3f}, "
            f"decode_locality={float(lora.get('decode_locality_rate', 0.0) or 0.0):.3f}, "
            f"max_fair_gap={float(lora.get('max_adapter_fairness_gap', 0.0) or 0.0):.3f}"
        )
        lines.append(
            "  RUNTIME(lora,current_adaptive): "
            f"admit_relaxed_rate={float(lora.get('admit_relaxed_rate', 0.0) or 0.0):.3f}, "
            f"prefill_tightened_rate={float(lora.get('prefill_tightened_rate', 0.0) or 0.0):.3f}, "
            f"decode_relaxed_rate={float(lora.get('decode_relaxed_rate', 0.0) or 0.0):.3f}"
        )
        per_adapter = lora.get("adapter_fairness_gap", {})
        if isinstance(per_adapter, dict) and per_adapter:
            formatted = ", ".join(
                f"{key}={float(value or 0.0):+.3f}"
                for key, value in sorted(per_adapter.items())
            )
            lines.append(f"  RUNTIME(lora,current_by_adapter): {formatted}")
    if isinstance(multimodal, dict):
        lines.append(
            "  RUNTIME(multimodal,current): "
            f"requests={float(multimodal.get('request_count', 0.0) or 0.0):.3f}, "
            f"mm_lora_requests={float(multimodal.get('multimodal_lora_request_count', 0.0) or 0.0):.3f}, "
            f"mm_lora_prefills={float(multimodal.get('multimodal_lora_prefill_count', 0.0) or 0.0):.3f}, "
            f"queued={float(multimodal.get('queued_request_count', 0.0) or 0.0):.3f}, "
            f"p95_wait={float(multimodal.get('p95_queue_wait_s', 0.0) or 0.0):.3f}s, "
            f"images_per_request={float(multimodal.get('images_per_request', 0.0) or 0.0):.3f}, "
            f"embed_rate={float(multimodal.get('embedding_compute_rate', 0.0) or 0.0):.3f}, "
            f"prefix_hit_rate={float(multimodal.get('prefix_cache_hit_rate', 0.0) or 0.0):.3f}, "
            f"saved_prefill_tok_per_req={float(multimodal.get('saved_prefill_tokens_per_request', 0.0) or 0.0):.3f}, "
            f"mm_prefill_limit={float(multimodal.get('avg_effective_prefill_multimodal_limit', 0.0) or 0.0):.3f}, "
            f"mm_prefill_relaxed_rate={float(multimodal.get('prefill_multimodal_limit_relaxed_rate', 0.0) or 0.0):.3f}, "
            f"mm_prefill_tightened_rate={float(multimodal.get('prefill_multimodal_limit_tightened_rate', 0.0) or 0.0):.3f}, "
            f"preempt_rate={float(multimodal.get('preempted_prefill_rate', 0.0) or 0.0):.3f}, "
            f"mm_lora_batch_ratio={float(multimodal.get('mixed_multimodal_lora_batch_ratio', 0.0) or 0.0):.3f}, "
            f"mm_lora_admit_limit={float(multimodal.get('avg_effective_admit_multimodal_lora_limit', 0.0) or 0.0):.3f}, "
            f"mm_lora_prefill_limit={float(multimodal.get('avg_effective_prefill_multimodal_lora_limit', 0.0) or 0.0):.3f}, "
            f"mm_lora_decode_limit={float(multimodal.get('avg_effective_decode_multimodal_lora_limit', 0.0) or 0.0):.3f}, "
            f"mm_lora_gap={float(multimodal.get('avg_prefill_multimodal_lora_max_fairness_gap', 0.0) or 0.0):.3f}, "
            f"mm_lora_decode_gap={float(multimodal.get('avg_decode_multimodal_lora_max_fairness_gap', 0.0) or 0.0):.3f}, "
            f"mm_lora_admit_trigger_rate={float(multimodal.get('admit_multimodal_lora_limit_triggered_rate', 0.0) or 0.0):.3f}, "
            f"mm_lora_prefill_trigger_rate={float(multimodal.get('prefill_multimodal_lora_limit_triggered_rate', 0.0) or 0.0):.3f}, "
            f"mm_lora_decode_trigger_rate={float(multimodal.get('decode_multimodal_lora_limit_triggered_rate', 0.0) or 0.0):.3f}, "
            f"mm_lora_prefill_relaxed_rate={float(multimodal.get('prefill_multimodal_lora_limit_relaxed_rate', 0.0) or 0.0):.3f}, "
            f"mm_lora_prefill_tightened_rate={float(multimodal.get('prefill_multimodal_lora_limit_tightened_rate', 0.0) or 0.0):.3f}, "
            f"mm_lora_fair_relax_rate={float(multimodal.get('prefill_multimodal_lora_limit_relaxed_by_fairness_rate', 0.0) or 0.0):.3f}, "
            f"mm_lora_local_tighten_rate={float(multimodal.get('prefill_multimodal_lora_limit_tightened_by_locality_rate', 0.0) or 0.0):.3f}, "
            f"mm_lora_decode_relaxed_rate={float(multimodal.get('decode_multimodal_lora_limit_relaxed_rate', 0.0) or 0.0):.3f}, "
            f"mm_lora_decode_tightened_rate={float(multimodal.get('decode_multimodal_lora_limit_tightened_rate', 0.0) or 0.0):.3f}, "
            f"mm_lora_decode_fair_relax_rate={float(multimodal.get('decode_multimodal_lora_limit_relaxed_by_fairness_rate', 0.0) or 0.0):.3f}, "
            f"mm_lora_decode_local_tighten_rate={float(multimodal.get('decode_multimodal_lora_limit_tightened_by_locality_rate', 0.0) or 0.0):.3f}"
        )
    return lines


def _derive_awq_metrics(stats: Dict[str, int]) -> Dict[str, float]:
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
) -> Dict[str, float]:
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
    deepseek_batched_engine: bool = False,
) -> Dict[str, Any]:
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
        return _run_deepseek_v4_flash_direct_benchmark(
            spec,
            batched_engine=deepseek_batched_engine,
        )

    if not os.path.isdir(spec.model_path) and not _is_hf_repo_id(spec.model_path):
        print("  [Skip] Model directory not found.")
        return {"skipped": 1.0}

    old_env = _apply_temp_env(spec.stable_env)
    llm: Optional[AsyncLLM] = None
    runtime_stats_by_phase: Dict[str, Dict[str, object]] = {}
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
        except asyncio.TimeoutError:
            awq_stats: Dict = {}
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
        awq_metrics: Dict[str, float] = {}
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
    decode_tps_aggregate = (
        decode_tokens_total * 1000.0 / decode_ms_total
        if decode_tokens_total > 0.0 and decode_ms_total > 0.0
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
        "ttft_p95_ms": (
            _p95(elapsed_ms_values) if elapsed_ms_values else float("nan")
        ),
        "e2e_p50_ms": (
            median(elapsed_ms_values) if elapsed_ms_values else float("nan")
        ),
        "e2e_p95_ms": (
            _p95(elapsed_ms_values) if elapsed_ms_values else float("nan")
        ),
        "prefill_p50_ms": float("nan"),
        "prefill_p95_ms": float("nan"),
        "decode_p50_ms": (
            median(elapsed_ms_values) if elapsed_ms_values else float("nan")
        ),
        "decode_p95_ms": (
            _p95(elapsed_ms_values) if elapsed_ms_values else float("nan")
        ),
        "prompt_tokens_total": 1.0,
        "prefill_tps_aggregate": 0.0,
        "prefill_tps_p50": float("nan"),
        "prefill_tps_p95": float("nan"),
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
            "kind": "deepseek_v4_flash_direct_gguf",
            "context_length": int(
                payload.get("context_length", spec.max_model_len)
            ),
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
                "usable_inference_metrics": payload.get(
                    "usable_inference_metrics", {}
                ),
            }
        },
        "warmup_trace": [],
    }


def _run_deepseek_v4_flash_direct_benchmark(
    spec: ModelSpec,
    *,
    batched_engine: bool = False,
) -> dict[str, Any]:
    if batched_engine:
        print("  [DeepSeek] Running batched-engine direct GGUF GPU smoke benchmark.")
    else:
        print("  [DeepSeek] Running direct GGUF GPU smoke benchmark.")
    if not os.path.isfile(spec.model_path):
        print("  [Skip] DeepSeek V4 Flash GGUF file not found.")
        return {"skipped": 1.0, "skip_reason": "model_file_not_found"}
    with tempfile.NamedTemporaryFile(
        prefix="fastinference_deepseek_v4_flash_",
        suffix=".json",
        delete=False,
    ) as tmp:
        profile_json = tmp.name
    script = (
        "tests/tools/run_deepseek_v4_flash_gpu_smoke_batched.py"
        if batched_engine
        else "tests/tools/run_deepseek_v4_flash_gpu_smoke.py"
    )
    command = [
        sys.executable,
        script,
        "--model",
        spec.model_path,
        "--context-length",
        str(spec.max_model_len),
        "--max-tokens",
        str(spec.max_new_tokens),
        "--profile-json",
        profile_json,
    ]
    if batched_engine:
        command.extend(
            [
                "--batch-size",
                str(spec.concurrent_reqs),
                "--prompt-length",
                "1",
                "--repeat",
                "1",
            ]
        )
    else:
        command.extend(
            [
                "--warmup-tokens",
                str(spec.max_new_tokens),
                "--repeat",
                "1",
            ]
        )
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
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
        with open(profile_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
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
    finally:
        try:
            os.unlink(profile_json)
        except OSError:
            pass


def _load_perf_baseline(path: str) -> dict[str, Any]:
    if not path.strip():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    summary = payload.get("summary", payload)
    return dict(summary) if isinstance(summary, dict) else {}


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end performance benchmark for Gemma4 26B/31B, Qwen3.5, "
            "TinyLlama, and DeepSeek V4 Flash Q2 GGUF direct-path models."
        )
    )
    parser.add_argument(
        "--models",
        type=str,
        default="gemma4_26b_a4b,gemma4_31b_q4,deepseek_v4_flash_q2_gguf",
        help=(
            "Comma-separated model keys. Default: "
            "gemma4_26b_a4b,gemma4_31b_q4,deepseek_v4_flash_q2_gguf. "
            "Available: tinyllama, qwen35_9b_awq, gemma4_31b_q4, "
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
    parser.add_argument(
        "--deepseek-batched-engine",
        action="store_true",
        default=False,
        help=(
            "For DeepSeek V4 Flash, run the batched engine method "
            "(generate_deepseek_v4_flash_greedy_batched) instead of one request at a time."
        ),
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
    specs: List[ModelSpec] = []
    resolved_scheduler_policy: Dict[str, Dict[str, Any]] = {}
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

    summary: Dict[str, Dict[str, Any]] = {}
    runtime_stats_summary: Dict[str, Dict[str, object]] = {}
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
                    deepseek_batched_engine=args.deepseek_batched_engine,
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


if __name__ == "__main__":
    asyncio.run(main())
