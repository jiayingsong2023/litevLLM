# SPDX-License-Identifier: Apache-2.0
import argparse
import asyncio
import copy
import json
import math
import os
import time
from dataclasses import dataclass, replace
from statistics import median
from typing import Dict, List, Optional

import torch

from vllm import SamplingParams
from vllm.config import CacheConfig, LoadConfig, ModelConfig, SchedulerConfig, VllmConfig
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


# KV cache: FP8 KV (FASTINFERENCE_KV_FP8=1) to save VRAM; aligned with accuracy suite defaults.
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
        stable_env={"FASTINFERENCE_KV_TYPE": "fp8"},
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
}

# ROCm/HIP: large BS × long prefill + AWQ Triton + FP8 KV often triggers hipErrorLaunchFailure
# (often async OOM or kernel fault). Use conservative defaults unless aggressive mode is on.
_ROCM_E2E_SAFE_CONCURRENT = 4
_ROCM_E2E_SAFE_PROMPT_TOKENS = 1024
_ROCM_E2E_SAFE_GPU_MEM_UTIL = 0.85


def _is_rocm() -> bool:
    return bool(getattr(torch.version, "hip", None))


def _is_e2e_aggressive(args: argparse.Namespace) -> bool:
    if getattr(args, "aggressive", False):
        return True
    raw = os.environ.get("FASTINFERENCE_E2E_AGGRESSIVE", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _maybe_apply_rocm_safe_profile(spec: ModelSpec, aggressive: bool) -> ModelSpec:
    """Downgrade batch/prompt on AMD ROCm unless aggressive mode is enabled."""
    if aggressive or not _is_rocm():
        return spec
    if spec.key not in ("tinyllama", "qwen35_9b_awq"):
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


def _read_awq_group_size_and_bits(model_path: str) -> tuple[int, int]:
    config_path = os.path.join(model_path, "config.json")
    group_size, bits = 128, 4
    try:
        if not os.path.isfile(config_path):
            return group_size, bits
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        qc = raw.get("quantization_config") or {}
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
        print(f"  [Warn] Failed to parse AWQ config ({config_path}): {exc}")
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


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(0.95 * (len(ordered) - 1))
    return ordered[idx]


def _finite_values(values: List[float]) -> List[float]:
    return [v for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]


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
    derived_metrics = _derive_runtime_metrics(snapshot)
    return {
        "phase": phase,
        "collected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stats": snapshot,
        "derived_metrics": derived_metrics,
    }


def _derive_runtime_metrics(snapshot: Dict[str, object]) -> Dict[str, object]:
    observer = snapshot.get("observer", {})
    backend = snapshot.get("backend", {})
    if not isinstance(observer, dict) or not isinstance(backend, dict):
        return {}

    observer_prefix = observer.get("prefix_cache", {})
    backend_prefix = backend.get("prefix_cache", {})
    observer_preemption = observer.get("preemption", {})
    observer_fairness = observer.get("fairness", {})
    if not isinstance(observer_prefix, dict):
        observer_prefix = {}
    if not isinstance(backend_prefix, dict):
        backend_prefix = {}
    if not isinstance(observer_preemption, dict):
        observer_preemption = {}
    if not isinstance(observer_fairness, dict):
        observer_fairness = {}

    request_count = float(observer_prefix.get("events", 0) or 0)
    step_count = float(observer.get("step_count", 0) or 0)
    admitted_requests = float(observer.get("admitted_requests", 0) or 0)
    materialized_hits = float(backend.get("prefix_cache_materialized_hits", 0) or 0)
    materialized_saved = float(
        backend.get("prefix_cache_materialized_saved_prefill_tokens", 0) or 0
    )
    lookup_comparisons = float(backend_prefix.get("lookup_comparisons", 0) or 0)
    lookup_candidates_total = float(backend_prefix.get("lookup_candidates_total", 0) or 0)
    preempted_steps = float(observer_preemption.get("preempted_steps", 0) or 0)
    preempted_prefills = float(
        observer_preemption.get("preempted_prefill_requests", 0) or 0
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

    return {
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
            "preempted_step_rate": (
                preempted_steps / step_count if step_count else 0.0
            ),
            "preempted_prefill_requests_per_step": (
                preempted_prefills / step_count if step_count else 0.0
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

    warmup_prefix = warmup_metrics.get("prefix_cache", {})
    benchmark_prefix = benchmark_metrics.get("prefix_cache", {})
    warmup_preemption = warmup_metrics.get("preemption", {})
    benchmark_preemption = benchmark_metrics.get("preemption", {})
    warmup_fairness = warmup_metrics.get("fairness", {})
    benchmark_fairness = benchmark_metrics.get("fairness", {})
    if not isinstance(warmup_prefix, dict) or not isinstance(benchmark_prefix, dict):
        warmup_prefix, benchmark_prefix = {}, {}
    if not isinstance(warmup_preemption, dict) or not isinstance(benchmark_preemption, dict):
        warmup_preemption, benchmark_preemption = {}, {}
    if not isinstance(warmup_fairness, dict) or not isinstance(benchmark_fairness, dict):
        warmup_fairness, benchmark_fairness = {}, {}

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
        "preempted_step_rate",
        "preempted_prefill_requests_per_step",
    )
    preemption_delta = {}
    for key in preemption_keys:
        preemption_delta[key] = float(benchmark_preemption.get(key, 0.0) or 0.0) - float(
            warmup_preemption.get(key, 0.0) or 0.0
        )

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

    return {
        "baseline_phase": "warmup",
        "target_phase": "benchmark",
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
    }


def _format_runtime_phase_diff_summary(
    phase_diffs: Dict[str, object],
) -> list[str]:
    if not isinstance(phase_diffs, dict):
        return []

    lines: list[str] = []
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
                f"prefills_per_step_delta={float(preemption_delta.get('preempted_prefill_requests_per_step', 0.0) or 0.0):+.3f}"
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

        group_size, bits = _read_awq_group_size_and_bits(spec.model_path)
        v_cfg.quant_config = AWQConfig(weight_bits=bits, group_size=group_size)
    return v_cfg


async def _run_single_request(
    llm: AsyncLLM,
    request_id: str,
    prompt: str,
    sampling_params: SamplingParams,
) -> Dict[str, float]:
    start = time.perf_counter()
    first_token_at: Optional[float] = None
    generated_tokens = 0
    async for output in llm.generate(prompt, sampling_params, request_id):
        if output.outputs:
            generated_tokens = len(output.outputs[0].token_ids)
            if first_token_at is None and generated_tokens > 0:
                first_token_at = time.perf_counter()
    end = time.perf_counter()
    ttft_ms = ((first_token_at - start) * 1000.0) if first_token_at is not None else float("nan")
    e2e_ms = (end - start) * 1000.0
    decode_tokens = max(0, generated_tokens - 1)
    decode_ms = e2e_ms - ttft_ms if math.isfinite(ttft_ms) else float("nan")
    decode_tps = (
        (float(decode_tokens) * 1000.0 / decode_ms)
        if decode_tokens > 0 and math.isfinite(decode_ms) and decode_ms > 0.0
        else float("nan")
    )
    return {
        "tokens": float(generated_tokens),
        "ttft_ms": ttft_ms,
        "e2e_ms": e2e_ms,
        "decode_tokens": float(decode_tokens),
        "decode_ms": decode_ms,
        "decode_tps": decode_tps,
    }


async def run_benchmark(spec: ModelSpec) -> Dict[str, float]:
    print(f"\n{'=' * 72}")
    print(f"BENCHMARKING: {spec.display_name}")
    print(f"PATH: {spec.model_path}")
    print(
        "SETUP: "
        f"concurrency={spec.concurrent_reqs}, prompt_tokens~{spec.prompt_tokens_target}, "
        f"max_new_tokens={spec.max_new_tokens}, quant={spec.quant}"
    )
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(
            "GPU MEMORY: "
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated / {total_mem:.2f} GB total"
        )
    print(f"{'=' * 72}")

    if not os.path.isdir(spec.model_path):
        print("  [Skip] Model directory not found.")
        return {"skipped": 1.0}

    old_env = _apply_temp_env(spec.stable_env)
    llm: Optional[AsyncLLM] = None
    runtime_stats_by_phase: Dict[str, Dict[str, object]] = {}
    try:
        if spec.quant == "awq":
            from vllm.model_executor.layers.quantization.tensor import (
                get_awq_runtime_stats,
                reset_awq_runtime_stats,
            )

            reset_awq_runtime_stats()
        v_cfg = _build_vllm_config(spec)
        llm = AsyncLLM.from_vllm_config(v_cfg)
        sampling_params = SamplingParams(max_tokens=spec.max_new_tokens, temperature=0.0)
        # Reserve decode room so prefill does not consume the full KV capacity.
        prompt_budget = min(
            int(spec.prompt_tokens_target),
            max(8, int(spec.max_model_len) - int(spec.max_new_tokens) - 1),
        )
        prompt = _build_prompt(llm.engine.tokenizer, prompt_budget)

        print("  [Warmup] Running one short request...")
        async for _ in llm.generate("Hello", SamplingParams(max_tokens=8, temperature=0.0), f"{spec.key}_warmup"):
            pass
        runtime_stats_by_phase["warmup"] = _collect_runtime_stats(llm, phase="warmup")
        llm.reset_stats(clear_prefix_cache=False)

        print("  [Run] Launching concurrent benchmark requests...")
        wall_start = time.perf_counter()
        tasks = [
            _run_single_request(
                llm=llm,
                request_id=f"{spec.key}_{idx}_{int(time.time())}",
                prompt=prompt,
                sampling_params=sampling_params,
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
            if spec.quant == "awq":
                from vllm.model_executor.layers.quantization.tensor import (
                    get_awq_runtime_stats,
                )

                awq_stats = get_awq_runtime_stats()
            print(
                f"  [Timeout] Exceeded {spec.max_run_seconds}s for {spec.display_name}; "
                "marking as timeout in summary."
            )
            if awq_stats:
                print(f"  AWQ_STATS(timeout): {json.dumps(awq_stats, ensure_ascii=True, sort_keys=True)}")
            awq_metrics = _derive_awq_metrics(awq_stats) if awq_stats else {}
            benchmark_stats = _collect_runtime_stats(llm, phase="benchmark")
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
                "decode_tokens_total": 0.0,
                "decode_tps_aggregate": 0.0,
                "decode_tps_p50": float("nan"),
                "decode_tps_p95": float("nan"),
                "awq_runtime_stats": awq_stats,
                "awq_metrics": awq_metrics,
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
            }
        wall_end = time.perf_counter()
        runtime_stats_by_phase["benchmark"] = _collect_runtime_stats(llm, phase="benchmark")
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
        decode_tokens_total = float(sum(r["decode_tokens"] for r in request_results))
        # Sum of per-request decode_ms overlaps in time when concurrent_reqs>1, so this
        # denominator is not wall-clock decode duration — use aggregate_tps or decode_tps_p50.
        decode_tps_aggregate = (
            decode_tokens_total / (sum(decode_ms_list) / 1000.0)
            if decode_tokens_total > 0.0 and decode_ms_list and sum(decode_ms_list) > 0.0
            else 0.0
        )
        awq_stats = {}
        awq_metrics: Dict[str, float] = {}
        if spec.quant == "awq":
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
            "decode_tokens_total": decode_tokens_total,
            "decode_tps_aggregate": decode_tps_aggregate,
            "decode_tps_p50": median(decode_tps_list) if decode_tps_list else float("nan"),
            "decode_tps_p95": _p95(decode_tps_list) if decode_tps_list else float("nan"),
            "awq_runtime_stats": awq_stats,
            "awq_metrics": awq_metrics,
            "runtime_stats": runtime_stats_by_phase,
        }

        print(
            "  RESULT: "
            f"tokens/s={result['aggregate_tps']:.2f}, "
            f"TTFT p50/p95={result['ttft_p50_ms']:.1f}/{result['ttft_p95_ms']:.1f} ms, "
            f"E2E p50/p95={result['e2e_p50_ms']:.1f}/{result['e2e_p95_ms']:.1f} ms, "
            f"Decode p50/p95={result['decode_p50_ms']:.1f}/{result['decode_p95_ms']:.1f} ms, "
            f"Decode TPS(agg)={result['decode_tps_aggregate']:.2f}, "
            f"Decode TPS p50={result['decode_tps_p50']:.2f}"
        )
        if spec.concurrent_reqs > 1:
            print(
                "  [Note] Decode TPS(agg)=total_decode_tokens/sum(decode_ms); with concurrent "
                "requests those durations overlap, so this is not wall batch decode TPS. "
                "Prefer tokens/s (aggregate_tps) or decode_tps_p50 for A/B vs kernel work."
            )
        for line in _format_runtime_phase_diff_summary(
            runtime_stats_by_phase.get("phase_diffs", {})
        ):
            print(line)
        if awq_stats:
            print(f"  AWQ_STATS: {json.dumps(awq_stats, ensure_ascii=True, sort_keys=True)}")
        return result
    finally:
        if llm is not None:
            llm.shutdown()
        try:
            from vllm.model_executor.layers.quantization.tensor import clear_global_weight_cache

            clear_global_weight_cache()
        except Exception:
            pass
        _restore_env(old_env)
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end performance benchmark for TinyLlama / Qwen3.5 AWQ models."
    )
    parser.add_argument(
        "--models",
        type=str,
        default="tinyllama,qwen35_9b_awq",
        help="Comma-separated model keys: tinyllama,qwen35_9b_awq",
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
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    aggressive = _is_e2e_aggressive(args)
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
    for key in model_keys:
        if key not in MODEL_SPECS:
            raise ValueError(f"Unknown model key: {key}. Supported: {', '.join(MODEL_SPECS.keys())}")
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
        specs.append(spec)

    print("=" * 72)
    print("FASTINFERENCE END-TO-END PERFORMANCE REGRESSION")
    print("Targets: TinyLlama + Qwen3.5-9B AWQ")
    print("=" * 72)

    summary: Dict[str, Dict[str, float]] = {}
    runtime_stats_summary: Dict[str, Dict[str, object]] = {}
    for spec in specs:
        summary[spec.key] = await run_benchmark(spec)
        runtime_stats_summary[spec.key] = dict(
            summary[spec.key].get("runtime_stats", {})
        )

    print("\n" + "-" * 72)
    print("PERF SUMMARY")
    print(
        "(decode_tps = aggregate formula; with concurrent_reqs>1 it is not wall batch TPS — "
        "see decode_tps_p50 or aggregate_tps)"
    )
    print("-" * 72)
    for key in model_keys:
        r = summary[key]
        if r.get("skipped", 0.0) == 1.0:
            print(f"{key:16} | skipped")
            continue
        if r.get("timed_out", 0.0) == 1.0:
            print(f"{key:16} | timeout")
            continue
        print(
            f"{key:16} | tps={r['aggregate_tps']:.2f} | "
            f"ttft_p50={r['ttft_p50_ms']:.1f}ms | ttft_p95={r['ttft_p95_ms']:.1f}ms | "
            f"e2e_p50={r['e2e_p50_ms']:.1f}ms | e2e_p95={r['e2e_p95_ms']:.1f}ms | "
            f"decode_tps_agg={r['decode_tps_aggregate']:.2f} | "
            f"decode_tps_p50={r['decode_tps_p50']:.2f}"
        )

    if args.json_out:
        payload = {
            "models": model_keys,
            "summary": summary,
            "runtime_stats": runtime_stats_summary,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
        print(f"\nJSON summary written to: {args.json_out}")

    if args.runtime_stats_out:
        payload = {
            "models": model_keys,
            "runtime_stats": runtime_stats_summary,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(args.runtime_stats_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
        print(f"Runtime stats written to: {args.runtime_stats_out}")


if __name__ == "__main__":
    asyncio.run(main())
