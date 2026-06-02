# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from tests.tools.analyze_gemma4_mlp_streaming_design import analyze_designs


def test_mlp_streaming_design_costs_reject_naive_recompute() -> None:
    report = analyze_designs(hidden=5376, intermediate=21504, block_n=64, block_i=16)
    current = report["current_two_stage"]
    naive = report["naive_recompute_single_kernel"]

    assert current["gate_up_read_multiplier"] == 1
    assert naive["gate_up_read_multiplier"] == 84
    assert naive["estimated_bytes"] > current["estimated_bytes"] * 20
    assert naive["decision"] == "reject"


def test_mlp_streaming_design_costs_reject_partial_tile_staging() -> None:
    report = analyze_designs(hidden=5376, intermediate=21504, block_n=64, block_i=16)
    current = report["current_two_stage"]
    partial = report["intermediate_tile_partial_staging"]

    assert partial["partial_bytes"] == 5376 * (21504 // 16) * 4
    assert partial["partial_bytes"] > current["activation_bytes"] * 100
    assert partial["decision"] == "reject"


def test_mlp_streaming_design_requires_true_shared_intermediate_candidate() -> None:
    report = analyze_designs(hidden=5376, intermediate=21504, block_n=64, block_i=16)
    shared = report["true_shared_intermediate_required"]

    assert shared["gate_up_read_multiplier"] == 1
    assert shared["partial_bytes"] == 0
    assert shared["decision"] == "requires_cross_program_sharing"
