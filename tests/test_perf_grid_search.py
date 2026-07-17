# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    p = root / "tests" / "tools" / "perf_grid_search.py"
    spec = importlib.util.spec_from_file_location("perf_grid_search", p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_coarse_grid_has_expected_shape() -> None:
    mod = _load_module()
    grid = mod.build_coarse_grid()
    assert len(grid) == 8
    assert grid[0].prefill_chunk == 256
    assert grid[0].prefill_microbatch == 2
    assert all(cfg.kv_type == "turbo_int4" for cfg in grid)


def test_fine_grid_deduplicates_by_core_knobs() -> None:
    mod = _load_module()
    base = mod.GridConfig(
        name="x",
        prefill_chunk=256,
        prefill_microbatch=2,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=2,
        prefill_catchup_ratio=0.25,
    )
    fine = mod.build_fine_grid([base, base])
    assert len(fine) == 4
    tags = {(x.prefill_reserved_tokens, x.prefill_reserve_backlog) for x in fine}
    assert tags == {(0, 1), (0, 2), (128, 1), (128, 2)}


def test_select_top_prefers_decode_tps_then_ttft() -> None:
    mod = _load_module()
    rows = [
        {
            "name": "a",
            "status": "pass",
            "decode_tps_aggregate": 10.0,
            "ttft_p50_ms": 1000.0,
        },
        {
            "name": "b",
            "status": "pass",
            "decode_tps_aggregate": 11.0,
            "ttft_p50_ms": 2000.0,
        },
        {
            "name": "c",
            "status": "pass",
            "decode_tps_aggregate": 11.0,
            "ttft_p50_ms": 1500.0,
        },
        {
            "name": "d",
            "status": "timed_out",
            "decode_tps_aggregate": 99.0,
            "ttft_p50_ms": 1.0,
        },
    ]
    top = mod._select_top(rows, 2)
    assert [x["name"] for x in top] == ["c", "b"]
