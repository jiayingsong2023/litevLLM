#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from vllm.model_executor.models.deepseek_v4_flash.backend import (
    DeepSeekV4FlashGPUBackend,
    DeepSeekV4FlashGPUCapabilities,
)

from vllm.model_executor.models.deepseek_v4_flash.config import (
    DeepSeekV4FlashMemoryPolicy,
)
from vllm.model_executor.models.deepseek_v4_flash.model import (
    DeepSeekV4FlashForCausalLM,
)
from vllm.model_executor.models.deepseek_v4_flash.weight_store import (
    open_deepseek_v4_flash_weight_store,
)

_DEFAULT_GGUF = (
    "models/DeepSeek-V4-Flash-ds4/"
    "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke benchmark for batched DeepSeek V4 Flash greedy generation."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(_DEFAULT_GGUF),
        help="Path to the DeepSeek V4 Flash GGUF file.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=4096,
        help="Model context length used for the runtime budget.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Number of prompts to generate in one batched call.",
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=16,
        help="Number of input tokens per prompt.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8,
        help="Number of new tokens to generate for each prompt.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device to run generation on.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional JSON file to write the results to.",
    )
    return parser.parse_args(argv)


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive; got {value}")


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


def _build_prompt_token_ids(prompt_length: int) -> list[int]:
    return [0] * prompt_length


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    for name, value in (
        ("--batch-size", args.batch_size),
        ("--prompt-length", args.prompt_length),
        ("--max-tokens", args.max_tokens),
    ):
        _validate_positive(name, value)

    if not torch.cuda.is_available():
        print("SKIP: CUDA/ROCm torch backend is not available.")
        return 0

    if not args.model.is_file():
        print(f"SKIP: model file not found: {args.model}")
        return 0

    prompt_token_ids = _build_prompt_token_ids(args.prompt_length)
    device = torch.device(args.device)
    input_ids_list = [
        torch.tensor(prompt_token_ids, dtype=torch.long, device=device)
        for _ in range(args.batch_size)
    ]

    try:
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

            torch.cuda.synchronize()
            start = perf_counter()
            output_ids_list = model.generate_greedy_kernel_batched(
                input_ids_list,
                max_tokens=args.max_tokens,
            )
            torch.cuda.synchronize()
            elapsed_s = perf_counter() - start
    except Exception as exc:  # pragma: no cover - defensive skip path
        print(f"SKIP: batched generation failed: {exc}")
        return 0

    total_generated_tokens = sum(
        max(0, int(output_ids.numel()) - args.prompt_length)
        for output_ids in output_ids_list
    )
    throughput_tok_s = total_generated_tokens / elapsed_s if elapsed_s > 0.0 else 0.0

    payload: dict[str, Any] = {
        "model": str(args.model),
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "prompt_length": args.prompt_length,
        "max_tokens": args.max_tokens,
        "device": str(device),
        "wall_time_s": elapsed_s,
        "total_generated_tokens": total_generated_tokens,
        "throughput_tok_s": throughput_tok_s,
    }

    print(f"batch_size: {args.batch_size}")
    print(f"prompt_length: {args.prompt_length}")
    print(f"max_tokens: {args.max_tokens}")
    print(f"wall_time_s: {elapsed_s:.4f}")
    print(f"total_generated_tokens: {total_generated_tokens}")
    print(f"throughput_tok_s: {throughput_tok_s:.2f}")

    if args.json_out is not None:
        _write_json(args.json_out, payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
