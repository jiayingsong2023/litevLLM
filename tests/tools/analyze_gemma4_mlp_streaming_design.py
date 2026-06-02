# SPDX-License-Identifier: Apache-2.0
"""Cost model for Gemma4-31B M=1 AWQ MLP streaming candidates."""

from __future__ import annotations

import argparse
import json
from typing import Any


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def analyze_designs(
    *,
    hidden: int = 5376,
    intermediate: int = 21504,
    block_n: int = 64,
    block_i: int = 16,
) -> dict[str, dict[str, Any]]:
    """Return approximate traffic and work multipliers for P2c designs.

    The model intentionally tracks only first-order decode costs that decide
    whether a design deserves a Triton implementation:
      - packed INT4 gate/up/down qweight bytes
      - scale bytes, approximated as fp16 scales
      - activation/intermediate bytes
      - float32 partial staging bytes
      - gate/up recompute multiplier across down output tiles
    """
    if hidden % 32 != 0 or intermediate % 32 != 0:
        raise ValueError("hidden and intermediate must be group32 aligned")
    if block_n <= 0 or block_i <= 0:
        raise ValueError("block_n and block_i must be positive")

    hidden_groups = hidden // 32
    intermediate_groups = intermediate // 32
    n_tiles = _ceil_div(hidden, block_n)
    i_tiles = _ceil_div(intermediate, block_i)

    gate_up_qweight_bytes = 2 * intermediate * hidden // 2
    down_qweight_bytes = hidden * intermediate // 2
    gate_up_scale_bytes = 2 * intermediate * hidden_groups * 2
    down_scale_bytes = hidden * intermediate_groups * 2
    activation_bytes = intermediate * 2
    current_bytes = (
        gate_up_qweight_bytes
        + down_qweight_bytes
        + gate_up_scale_bytes
        + down_scale_bytes
        + activation_bytes
        + activation_bytes * n_tiles
    )

    naive_bytes = (
        (gate_up_qweight_bytes + gate_up_scale_bytes) * n_tiles
        + down_qweight_bytes
        + down_scale_bytes
    )
    partial_bytes = hidden * i_tiles * 4
    partial_staging_bytes = (
        gate_up_qweight_bytes
        + gate_up_scale_bytes
        + down_qweight_bytes
        + down_scale_bytes
        + partial_bytes
        + partial_bytes
    )

    return {
        "current_two_stage": {
            "estimated_bytes": current_bytes,
            "gate_up_read_multiplier": 1,
            "activation_bytes": activation_bytes,
            "partial_bytes": 0,
            "decision": "baseline",
        },
        "naive_recompute_single_kernel": {
            "estimated_bytes": naive_bytes,
            "gate_up_read_multiplier": n_tiles,
            "activation_bytes": 0,
            "partial_bytes": 0,
            "decision": "reject",
            "reason": "recomputes gate/up for each down output tile",
        },
        "intermediate_tile_partial_staging": {
            "estimated_bytes": partial_staging_bytes,
            "gate_up_read_multiplier": 1,
            "activation_bytes": 0,
            "partial_bytes": partial_bytes,
            "decision": "reject",
            "reason": "float32 partial staging dwarfs activation traffic",
        },
        "true_shared_intermediate_required": {
            "estimated_bytes": current_bytes - activation_bytes * n_tiles,
            "gate_up_read_multiplier": 1,
            "activation_bytes": 0,
            "partial_bytes": 0,
            "decision": "requires_cross_program_sharing",
            "reason": "needs gate/up values shared across down output tiles",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=5376)
    parser.add_argument("--intermediate", type=int, default=21504)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--block-i", type=int, default=16)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()
    payload = analyze_designs(
        hidden=args.hidden,
        intermediate=args.intermediate,
        block_n=args.block_n,
        block_i=args.block_i,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")


if __name__ == "__main__":
    main()
