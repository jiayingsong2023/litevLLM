#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from vllm.engine.lite_engine import LiteEngine
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
from vllm.sampling_params import SamplingParams


class _FakeTokenizer:
    """Minimal tokenizer for batched-engine throughput measurement.

    Encodes a prompt to a fixed token id repeated by prompt length, and decodes
    generated ids to an empty string. The benchmark only cares about token
    throughput, not text quality.
    """

    def __init__(self, prompt_token_id: int) -> None:
        self.eos_token_id = 0
        self.prompt_token_id = prompt_token_id

    def encode(self, prompt: str) -> list[int]:
        return [self.prompt_token_id] * len(prompt)

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        del skip_special_tokens
        return ""


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
        description="Run DeepSeek V4 Flash batched-engine GPU smoke."
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
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--prompt-length", type=int, default=1)
    parser.add_argument("--prompt-token-id", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
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


def _make_fake_engine(model: DeepSeekV4FlashForCausalLM) -> LiteEngine:
    engine = object.__new__(LiteEngine)
    engine.device = torch.device("cuda")
    engine.model = model
    engine.tokenizer = _FakeTokenizer(1)
    engine._deepseek_v4_flash_direct = True
    return engine


def main() -> int:
    args = parse_args()
    if args.repeat <= 0:
        raise SystemExit("--repeat must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.prompt_length <= 0:
        raise SystemExit("--prompt-length must be positive")
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive")
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
        engine = _make_fake_engine(model)

        prompts = ["x" * args.prompt_length for _ in range(args.batch_size)]
        request_ids = [f"req-{i}" for i in range(args.batch_size)]
        sampling_params = SamplingParams(
            n=1,
            max_tokens=args.max_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
        )
        sampling_params_list = [sampling_params] * args.batch_size

        model.warm_decode_static_weights(engine.device)
        model.warm_decode_token_experts(
            torch.tensor([args.prompt_token_id], dtype=torch.long, device=engine.device)
        )
        # Warmup batched path once.
        engine.generate_deepseek_v4_flash_greedy_batched(
            request_ids=request_ids,
            prompts=prompts,
            sampling_params_list=sampling_params_list,
        )

        model.enable_deepseek_profile(args.profile_json is not None)
        runs: list[dict[str, Any]] = []
        output_cpu: list[list[int]] = []
        for run_idx in range(args.repeat):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_wall = perf_counter()
            start_event.record()
            outputs = engine.generate_deepseek_v4_flash_greedy_batched(
                request_ids=request_ids,
                prompts=prompts,
                sampling_params_list=sampling_params_list,
            )
            end_event.record()
            torch.cuda.synchronize()
            wall_elapsed_ms = (perf_counter() - start_wall) * 1000.0
            cuda_elapsed_ms = float(start_event.elapsed_time(end_event))
            elapsed_ms = cuda_elapsed_ms if cuda_elapsed_ms > 0.0 else wall_elapsed_ms

            output_cpu = [output.outputs[0].token_ids for output in outputs]
            for generated_ids in output_cpu:
                if len(generated_ids) != args.max_tokens:
                    raise RuntimeError(
                        "DeepSeek V4 Flash batched smoke returned unexpected "
                        f"generated token count: got {len(generated_ids)}, "
                        f"expected {args.max_tokens}"
                    )

            generated_token_count = args.batch_size * args.max_tokens
            tokens_per_second = (
                generated_token_count * 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0
            )
            runs.append(
                {
                    "run_idx": run_idx,
                    "cuda_elapsed_ms": cuda_elapsed_ms,
                    "elapsed_ms": elapsed_ms,
                    "wall_elapsed_ms": wall_elapsed_ms,
                    "generated_token_count": generated_token_count,
                    "output_token_ids": output_cpu[-1] if output_cpu else [],
                    "decode_tokens_total": float(generated_token_count),
                    "decode_ms_total": elapsed_ms,
                    "decode_tps_steady_state": tokens_per_second,
                    "tokens_per_second": tokens_per_second,
                }
            )

        full_profile = model.deepseek_profile()
        profile = profile_payload(full_profile, include_events=args.profile_events)
        gpu_staging = model.gpu_staging_memory_stats()
        gpu_backend = model.gpu_backend.stats()
        summary: dict[str, Any] = {
            "model": str(args.model),
            "context_length": args.context_length,
            "batch_size": args.batch_size,
            "prompt_length": args.prompt_length,
            "max_tokens": args.max_tokens,
            "repeat": args.repeat,
            "output_token_ids": output_cpu[-1] if output_cpu else [],
            "runs": runs,
            "profile": profile,
            "gpu_staging": gpu_staging,
            "gpu_backend": gpu_backend,
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
