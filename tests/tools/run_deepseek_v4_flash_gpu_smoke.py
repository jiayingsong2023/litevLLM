#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from vllm.model_executor.models.deepseek_v4_flash.config import (
    DeepSeekV4FlashMemoryPolicy,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_backend import (
    DeepSeekV4FlashGPUBackend,
    DeepSeekV4FlashGPUCapabilities,
)
from vllm.model_executor.models.deepseek_v4_flash.model import (
    DeepSeekV4FlashForCausalLM,
)
from vllm.model_executor.models.deepseek_v4_flash.weight_store import (
    open_deepseek_v4_flash_weight_store,
)


def _build_ready_backend() -> DeepSeekV4FlashGPUBackend:
    return DeepSeekV4FlashGPUBackend(
        capabilities=DeepSeekV4FlashGPUCapabilities(
            q8_linear=True,
            attention=True,
            compressed_attention=True,
            cache_update=True,
            moe=True,
            output=True,
        )
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an opt-in DeepSeek V4 Flash real GGUF GPU smoke."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "models/DeepSeek-V4-Flash-ds4/"
            "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
        ),
        help="Path to the target DeepSeek V4 Flash GGUF file.",
    )
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--prompt-token-id", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--warmup-tokens", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--min-steady-decode-tps", type=float, default=0.0)
    parser.add_argument(
        "--use-graph",
        action="store_true",
        help="Enable the DeepSeek V4 Flash CUDA/HIP graph decode path.",
    )
    parser.add_argument("--profile-json", type=Path, default=None)
    parser.add_argument(
        "--profile-events",
        action="store_true",
        help="Include full profiler event records in the profile payload.",
    )
    return parser.parse_args(argv)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def decode_metrics_from_token_times(token_elapsed_ms: list[float]) -> dict[str, float]:
    if not token_elapsed_ms:
        return {
            "decode_tokens_total": 0.0,
            "decode_ms_total": 0.0,
            "decode_tps_agg": 0.0,
            "decode_tps_steady_state": 0.0,
        }
    total_ms = float(sum(token_elapsed_ms))
    steady_ms = (
        float(sum(token_elapsed_ms[1:])) if len(token_elapsed_ms) > 1 else total_ms
    )
    steady_tokens = max(len(token_elapsed_ms) - 1, 1)
    return {
        "decode_tokens_total": float(len(token_elapsed_ms)),
        "decode_ms_total": total_ms,
        "decode_tps_agg": (
            len(token_elapsed_ms) * 1000.0 / total_ms if total_ms > 0.0 else 0.0
        ),
        "decode_tps_steady_state": (
            steady_tokens * 1000.0 / steady_ms if steady_ms > 0.0 else 0.0
        ),
    }


def validate_steady_decode_tps(
    *,
    decode_tps_steady_state: float,
    min_steady_decode_tps: float,
) -> None:
    if min_steady_decode_tps <= 0.0:
        return
    if decode_tps_steady_state >= min_steady_decode_tps:
        return
    raise SystemExit(
        "steady-state decode TPS below threshold: "
        f"{decode_tps_steady_state:.2f} < {min_steady_decode_tps:.2f}"
    )


def profile_payload(
    profile: dict[str, object],
    *,
    include_events: bool,
) -> dict[str, object]:
    payload = {
        "enabled": bool(profile.get("enabled", False)),
        "counters": profile.get("counters", {}),
        "aggregate_by_name": profile.get("aggregate_by_name", {}),
    }
    if include_events:
        payload["events"] = profile.get("events", [])
    return payload


def profile_summary_payload(model: DeepSeekV4FlashForCausalLM) -> dict[str, object]:
    profiler = getattr(model, "_deepseek_profiler", None)
    compact_summary = getattr(profiler, "compact_summary", None)
    if callable(compact_summary):
        summary = compact_summary()
        if isinstance(summary, dict):
            return summary
    return {
        "top_events": [],
        "phase_totals_ms": {},
    }


def decode_profile_top_sections(
    profile_summary: dict[str, object],
    *,
    limit: int = 12,
) -> list[dict[str, object]]:
    top_events = profile_summary.get("top_events", [])
    if not isinstance(top_events, list):
        return []
    return [
        dict(event)
        for event in top_events[: max(limit, 0)]
        if isinstance(event, dict)
    ]


def generate_greedy_with_token_timings(
    model: DeepSeekV4FlashForCausalLM,
    input_ids: torch.Tensor,
    *,
    max_tokens: int,
    use_graph: bool,
) -> tuple[torch.Tensor, list[float]]:
    timed_generate = getattr(model, "generate_greedy_kernel_timed", None)
    if timed_generate is not None:
        output_ids, token_elapsed_ms = timed_generate(
            input_ids,
            max_tokens=max_tokens,
            use_graph=use_graph,
        )
        return output_ids, [float(ms) for ms in token_elapsed_ms]

    output_ids = input_ids
    token_elapsed_ms: list[float] = []
    for _token_idx in range(max_tokens):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        output_ids = model.generate_greedy_kernel(
            output_ids,
            max_tokens=1,
            use_graph=use_graph,
        )
        end_event.record()
        torch.cuda.synchronize()
        token_elapsed_ms.append(float(start_event.elapsed_time(end_event)))
    return output_ids, token_elapsed_ms


def phase3_metrics(
    *,
    profile: dict[str, object],
    gpu_staging: dict[str, object],
    gpu_backend: dict[str, object],
) -> dict[str, int]:
    counters = profile.get("counters", {})
    if not isinstance(counters, dict):
        counters = {}
    return {
        "full_resident_enabled": int(gpu_staging.get("full_resident_enabled", 0)),
        "fused_selected_expert_api_calls": int(
            gpu_backend.get("fused_selected_expert_api_calls", 0)
        ),
        "lru_evictions": int(gpu_staging.get("lru_evictions", 0)),
        "streamed_bytes": int(gpu_staging.get("streamed_bytes", 0)),
        "prefetch_hits": int(gpu_staging.get("prefetch_hits", 0)),
        "prefetch_misses": int(gpu_staging.get("prefetch_misses", 0)),
        "prefetch_failures": int(gpu_staging.get("prefetch_failures", 0))
        + int(counters.get("deepseek_prefetch_failures", 0)),
        "quantized_expert_calls": int(gpu_backend.get("quantized_expert_calls", 0)),
        "cpu_sync_points": int(counters.get("cpu_sync_points", 0)),
    }


def phase4_metrics(
    *,
    profile: dict[str, object],
    gpu_staging: dict[str, object],
    gpu_backend: dict[str, object],
) -> dict[str, int]:
    return {
        "q2_k_triton_calls": int(gpu_backend.get("q2_k_triton_calls", 0)),
        "iq2_xxs_triton_calls": int(gpu_backend.get("iq2_xxs_triton_calls", 0)),
        "iq2_xxs_gate_up_fused_calls": int(
            gpu_backend.get("iq2_xxs_gate_up_fused_calls", 0)
        ),
        "q2_iq2_reference_fallback_calls": int(
            gpu_backend.get("q2_iq2_reference_fallback_calls", 0)
        ),
    }


def usable_inference_metrics(
    *,
    profile: dict[str, object],
    gpu_staging: dict[str, object],
    gpu_backend: dict[str, object],
) -> dict[str, int]:
    counters = profile.get("counters", {})
    if not isinstance(counters, dict):
        counters = {}
    return {
        "full_resident_enabled": int(gpu_staging.get("full_resident_enabled", 0)),
        "fused_selected_expert_api_calls": int(
            gpu_backend.get("fused_selected_expert_api_calls", 0)
        ),
        "iq2_xxs_gate_up_fused_calls": int(
            gpu_backend.get("iq2_xxs_gate_up_fused_calls", 0)
        ),
        "lru_evictions": int(gpu_staging.get("lru_evictions", 0)),
        "pinned_entries": int(gpu_staging.get("pinned_entries", 0)),
        "prefetch_failures": int(gpu_staging.get("prefetch_failures", 0))
        + int(counters.get("deepseek_prefetch_failures", 0)),
        "prefetch_hits": int(gpu_staging.get("prefetch_hits", 0)),
        "prefetch_misses": int(gpu_staging.get("prefetch_misses", 0)),
        "q2_iq2_reference_fallback_calls": int(
            gpu_backend.get("q2_iq2_reference_fallback_calls", 0)
        ),
        "routed_expert_id_materializations": int(
            gpu_staging.get("routed_expert_id_materializations", 0)
        ),
        "streamed_bytes": int(gpu_staging.get("streamed_bytes", 0)),
    }


def main() -> int:
    args = parse_args()
    if args.repeat <= 0:
        raise SystemExit("--repeat must be positive")
    if args.warmup_tokens < 0:
        raise SystemExit("--warmup-tokens must be non-negative")
    if not args.model.is_file():
        raise SystemExit(f"DeepSeek V4 Flash GGUF file not found: {args.model}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm torch backend is not available")

    policy = DeepSeekV4FlashMemoryPolicy()
    with open_deepseek_v4_flash_weight_store(args.model) as store:
        budget = policy.estimate_runtime_budget(
            args.context_length,
            model_mmap_bytes=store.diagnostics.file_size_bytes,
        )
        policy.validate_runtime_budget(budget)
        model = DeepSeekV4FlashForCausalLM(
            weight_store=store,
            runtime_budget=budget,
            gpu_backend=_build_ready_backend(),
        )
        input_ids = torch.tensor(
            [args.prompt_token_id],
            dtype=torch.long,
            device="cuda",
        )
        model.warm_decode_static_weights(input_ids.device)
        model.warm_decode_token_experts(input_ids)
        if args.warmup_tokens > 0:
            model.generate_greedy_kernel(
                input_ids,
                max_tokens=args.warmup_tokens,
                use_graph=args.use_graph,
            )
        torch.cuda.synchronize()
        model.enable_deepseek_profile(args.profile_json is not None)
        runs: list[dict[str, object]] = []
        output_cpu: list[int] = []
        for run_idx in range(args.repeat):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_wall = perf_counter()
            start_event.record()
            output_ids, token_elapsed_ms = generate_greedy_with_token_timings(
                model,
                input_ids,
                max_tokens=args.max_tokens,
                use_graph=args.use_graph,
            )
            end_event.record()
            torch.cuda.synchronize()
            wall_elapsed_ms = (perf_counter() - start_wall) * 1000.0
            cuda_elapsed_ms = float(start_event.elapsed_time(end_event))
            elapsed_ms = cuda_elapsed_ms if cuda_elapsed_ms > 0.0 else wall_elapsed_ms
            output_cpu = [int(token) for token in output_ids.detach().cpu().tolist()]
            if len(output_cpu) != 1 + args.max_tokens:
                raise RuntimeError(
                    "DeepSeek V4 Flash smoke returned unexpected token count: "
                    f"got {len(output_cpu)}, expected {1 + args.max_tokens}"
                )
            generated_token_count = len(token_elapsed_ms)
            if generated_token_count != args.max_tokens:
                raise RuntimeError(
                    "DeepSeek V4 Flash smoke returned unexpected timing count: "
                    f"got {generated_token_count}, expected {args.max_tokens}"
                )
            decode_metrics = decode_metrics_from_token_times(token_elapsed_ms)
            validate_steady_decode_tps(
                decode_tps_steady_state=decode_metrics["decode_tps_steady_state"],
                min_steady_decode_tps=args.min_steady_decode_tps,
            )
            tokens_per_second = (
                args.max_tokens * 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0
            )
            runs.append(
                {
                    "run_idx": run_idx,
                    "cuda_elapsed_ms": cuda_elapsed_ms,
                    "elapsed_ms": elapsed_ms,
                    "generated_token_count": generated_token_count,
                    "output_token_ids": output_cpu,
                    "token_elapsed_ms": token_elapsed_ms,
                    "tokens_per_second": tokens_per_second,
                    "wall_elapsed_ms": wall_elapsed_ms,
                    **decode_metrics,
                }
            )
        full_profile = model.deepseek_profile()
        profile = profile_payload(full_profile, include_events=args.profile_events)
        profile_summary = profile_summary_payload(model)
        gpu_staging = model.gpu_staging_memory_stats()
        gpu_backend = model.gpu_backend.stats()
        summary = {
            "model": str(args.model),
            "context_length": args.context_length,
            "max_tokens": args.max_tokens,
            "warmup_tokens": args.warmup_tokens,
            "use_graph": args.use_graph,
            "output_token_ids": output_cpu,
            "repeat": args.repeat,
            "runs": runs,
            "profile": profile,
            "profile_summary": profile_summary,
            "decode_profile_top_sections": decode_profile_top_sections(
                profile_summary
            ),
            "gpu_staging": gpu_staging,
            "gpu_backend": gpu_backend,
            "phase3_metrics": phase3_metrics(
                profile=profile,
                gpu_staging=gpu_staging,
                gpu_backend=gpu_backend,
            ),
            "phase4_metrics": phase4_metrics(
                profile=profile,
                gpu_staging=gpu_staging,
                gpu_backend=gpu_backend,
            ),
            "usable_inference_metrics": usable_inference_metrics(
                profile=profile,
                gpu_staging=gpu_staging,
                gpu_backend=gpu_backend,
            ),
            "runtime_budget": {
                "resident_bytes": budget.resident_bytes,
                "available_headroom_bytes": budget.available_headroom_bytes,
                "min_system_headroom_bytes": budget.min_system_headroom_bytes,
            },
        }
    if args.profile_json is not None:
        write_json(args.profile_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
