#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vllm.model_executor.models.deepseek_v4_flash.gguf_reader import (
    DeepSeekV4FlashTensor,
    ggml_tensor_nbytes,
)
from vllm.model_executor.models.deepseek_v4_flash.weight_store import (
    DeepSeekV4FlashGroupedExpertTensors,
    DeepSeekV4FlashSemanticBindings,
    open_deepseek_v4_flash_weight_store,
)

_GGUF_BLOCK_COLUMNS = 256


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect DeepSeek V4 Flash grouped expert tensor shapes."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the DeepSeek V4 Flash GGUF file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum records to emit; 0 means no limit.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to also write the JSON records.",
    )
    return parser.parse_args(argv)


def expert_shape_record(
    *,
    layer_idx: int,
    projection: str,
    tensor: DeepSeekV4FlashTensor,
) -> dict[str, int | str]:
    if len(tensor.dims) != 3:
        raise ValueError(
            f"grouped expert tensor {tensor.name} must have dims "
            f"(input, output, expert_count); got {tensor.dims}"
        )
    input_size, output_size, expert_count = tensor.dims
    return {
        "layer_idx": layer_idx,
        "projection": projection,
        "tensor_name": tensor.name,
        "ggml_type": tensor.tensor_type,
        "rows": input_size,
        "columns": output_size,
        "expert_count": expert_count,
        "columns_blocks": output_size // _GGUF_BLOCK_COLUMNS,
        "nbytes_per_expert": ggml_tensor_nbytes(
            (input_size, output_size),
            tensor.tensor_type,
        ),
    }


def _iter_grouped_expert_records(
    bindings: DeepSeekV4FlashSemanticBindings,
) -> list[dict[str, int | str]]:
    records: list[dict[str, int | str]] = []
    for layer in bindings.layers:
        grouped_experts = layer.grouped_experts
        if grouped_experts is None:
            continue
        records.extend(_records_for_group(layer.layer_index, grouped_experts))
    return records


def _records_for_group(
    layer_idx: int,
    grouped_experts: DeepSeekV4FlashGroupedExpertTensors,
) -> list[dict[str, int | str]]:
    return [
        expert_shape_record(
            layer_idx=layer_idx,
            projection="gate",
            tensor=grouped_experts.gate,
        ),
        expert_shape_record(
            layer_idx=layer_idx,
            projection="up",
            tensor=grouped_experts.up,
        ),
        expert_shape_record(
            layer_idx=layer_idx,
            projection="down",
            tensor=grouped_experts.down,
        ),
    ]


def collect_expert_shape_records(
    *,
    model: Path,
    limit: int = 0,
) -> list[dict[str, int | str]]:
    if limit < 0:
        raise ValueError("--limit must be non-negative")
    with open_deepseek_v4_flash_weight_store(model) as store:
        records = _iter_grouped_expert_records(store.bindings)
    if limit:
        return records[:limit]
    return records


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    records = collect_expert_shape_records(model=args.model, limit=args.limit)
    output = json.dumps(records, indent=2, sort_keys=True)
    print(output)
    if args.json is not None:
        write_json(args.json, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
