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


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
        output_ids = model.generate_greedy_kernel(
            input_ids,
            max_tokens=args.max_tokens,
        )
        output_cpu = [int(token) for token in output_ids.detach().cpu().tolist()]
        if len(output_cpu) != 1 + args.max_tokens:
            raise RuntimeError(
                "DeepSeek V4 Flash smoke returned unexpected token count: "
                f"got {len(output_cpu)}, expected {1 + args.max_tokens}"
            )
        summary = {
            "model": str(args.model),
            "context_length": args.context_length,
            "max_tokens": args.max_tokens,
            "output_token_ids": output_cpu,
            "gpu_staging": model.gpu_staging_memory_stats(),
            "runtime_budget": {
                "resident_bytes": budget.resident_bytes,
                "available_headroom_bytes": budget.available_headroom_bytes,
                "min_system_headroom_bytes": budget.min_system_headroom_bytes,
            },
        }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
