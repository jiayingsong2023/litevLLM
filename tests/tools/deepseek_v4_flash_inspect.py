#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

from vllm.model_executor.models.deepseek_v4_flash.config import (
    DeepSeekV4FlashMemoryPolicy,
)
from vllm.model_executor.models.deepseek_v4_flash.weight_store import (
    open_deepseek_v4_flash_weight_store,
)


def _format_tensor_type_counts(counts: dict[int, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(
        f"{tensor_type}={count}" for tensor_type, count in sorted(counts.items())
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()

    policy = DeepSeekV4FlashMemoryPolicy()
    with open_deepseek_v4_flash_weight_store(args.model) as store:
        model = store.model
        diagnostics = store.diagnostics
        print(f"model: {model.name}")
        print(f"layers: {model.shape.num_layers}")
        print(f"hidden: {model.shape.hidden_size}")
        print(f"vocab: {model.shape.vocab_size}")
        print(f"tensors: {diagnostics.tensor_count}")
        print(f"bound tensors: {diagnostics.bound_tensor_count}")
        tensor_type_counts = _format_tensor_type_counts(
            diagnostics.tensor_type_counts
        )
        print(f"tensor types: {tensor_type_counts}")
        for tensor_type, samples in sorted(diagnostics.tensor_type_samples.items()):
            for sample in samples:
                print(
                    f"type {tensor_type} sample: {sample.name} "
                    f"dims={sample.dims} offset={sample.offset}"
                )
        if diagnostics.unaligned_tensor_offsets:
            unaligned = ", ".join(diagnostics.unaligned_tensor_offsets)
        else:
            unaligned = "none"
        print(f"unaligned tensor offsets: {unaligned}")
        for ctx in (4096, 8192):
            budget = policy.estimate_runtime_budget(
                ctx,
                model_mmap_bytes=diagnostics.file_size_bytes,
            )
            print(f"context {ctx}: {budget.context.total_bytes} bytes")
            print(f"  model mmap bytes: {budget.model_mmap_bytes}")
            print(f"  resident bytes: {budget.resident_bytes}")
            print(f"  available UMA headroom: {budget.available_headroom_bytes}")


if __name__ == "__main__":
    main()
