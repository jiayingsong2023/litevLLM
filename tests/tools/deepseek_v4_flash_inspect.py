#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

from vllm.model_executor.models.deepseek_v4_flash.config import (
    DeepSeekV4FlashMemoryPolicy,
)
from vllm.model_executor.models.deepseek_v4_flash.weight_store import (
    DeepSeekV4FlashTensor,
    DeepSeekV4FlashWeightStore,
    open_deepseek_v4_flash_weight_store,
)


def _format_tensor_type_counts(counts: dict[int, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(
        f"{tensor_type}={count}" for tensor_type, count in sorted(counts.items())
    )


def _format_optional_tensor(tensor: DeepSeekV4FlashTensor | None) -> str:
    if tensor is None:
        return "missing"
    return tensor.name


def _print_semantic_binding_summary(store: DeepSeekV4FlashWeightStore) -> None:
    bindings = store.bindings
    print(
        "semantic outputs: "
        f"norm={_format_optional_tensor(bindings.output_norm)}, "
        f"head={_format_optional_tensor(bindings.output_head)}"
    )
    first_layer = bindings.layers[0]
    print(
        "semantic layer 0 attention: "
        f"query={_format_optional_tensor(first_layer.attention_query)}, "
        f"query_a={_format_optional_tensor(first_layer.attention_query_a)}, "
        f"query_b={_format_optional_tensor(first_layer.attention_query_b)}, "
        f"key_value={_format_optional_tensor(first_layer.attention_key_value)}, "
        f"output={_format_optional_tensor(first_layer.attention_output)}, "
        f"output_a={_format_optional_tensor(first_layer.attention_output_a)}, "
        f"output_b={_format_optional_tensor(first_layer.attention_output_b)}"
    )
    router_layers = sum(1 for layer in bindings.layers if layer.router is not None)
    routed_expert_layers = sum(1 for layer in bindings.layers if layer.routed_experts)
    grouped_expert_layers = sum(
        1 for layer in bindings.layers if layer.grouped_experts is not None
    )
    print(f"semantic layers with router bindings: {router_layers}")
    print(f"semantic layers with routed expert bindings: {routed_expert_layers}")
    print(f"semantic layers with grouped expert bindings: {grouped_expert_layers}")


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
        _print_semantic_binding_summary(store)
        tensor_type_counts = _format_tensor_type_counts(diagnostics.tensor_type_counts)
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
