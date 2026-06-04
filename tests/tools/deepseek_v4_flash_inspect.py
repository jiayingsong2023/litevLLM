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
        for ctx in (4096, 8192):
            estimate = policy.estimate_context_bytes(ctx)
            print(f"context {ctx}: {estimate.total_bytes} bytes")


if __name__ == "__main__":
    main()
