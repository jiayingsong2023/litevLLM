#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--profile-json", type=Path, default=None)
    return parser.parse_args(argv)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
        "q2_iq2_reference_fallback_calls": int(
            gpu_backend.get("q2_iq2_reference_fallback_calls", 0)
        ),
    }


def main() -> int:
    args = parse_args()
    if args.repeat <= 0:
        raise SystemExit("--repeat must be positive")
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
        model.enable_deepseek_profile(args.profile_json is not None)
        input_ids = torch.tensor(
            [args.prompt_token_id],
            dtype=torch.long,
            device="cuda",
        )
        runs: list[dict[str, object]] = []
        output_cpu: list[int] = []
        for run_idx in range(args.repeat):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            output_ids = model.generate_greedy_kernel(
                input_ids,
                max_tokens=args.max_tokens,
            )
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = float(start_event.elapsed_time(end_event))
            output_cpu = [
                int(token) for token in output_ids.detach().cpu().tolist()
            ]
            if len(output_cpu) != 1 + args.max_tokens:
                raise RuntimeError(
                    "DeepSeek V4 Flash smoke returned unexpected token count: "
                    f"got {len(output_cpu)}, expected {1 + args.max_tokens}"
                )
            tokens_per_second = (
                args.max_tokens * 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0
            )
            runs.append(
                {
                    "run_idx": run_idx,
                    "elapsed_ms": elapsed_ms,
                    "output_token_ids": output_cpu,
                    "tokens_per_second": tokens_per_second,
                }
            )
        profile = model.deepseek_profile()
        gpu_staging = model.gpu_staging_memory_stats()
        gpu_backend = model.gpu_backend.stats()
        summary = {
            "model": str(args.model),
            "context_length": args.context_length,
            "max_tokens": args.max_tokens,
            "output_token_ids": output_cpu,
            "repeat": args.repeat,
            "runs": runs,
            "profile": profile,
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
