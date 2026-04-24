# SPDX-License-Identifier: Apache-2.0
"""
Heuristic guardrail for ``_select_split_k``.

These tests pin the split-K policy across the shape families we care about
(Gemma4-31B gate/up/down projections, Qwen3.5-9B MLP, generic prefill) so
future tuning doesn't accidentally regress one model to help another.

The kernel itself is exercised by ``test_awq_fused_gemm_numerics.py``;
here we only validate the policy function, which is pure CPU code.
"""
from __future__ import annotations

import os
from typing import Iterator
from unittest import mock

import pytest

from vllm.kernels.triton.awq_fused_gemm import _select_split_k


@pytest.fixture(autouse=True)
def _clear_env() -> Iterator[None]:
    """Ensure each test sees the ``auto`` default without inherited state."""
    prior = os.environ.pop("FASTINFERENCE_AWQ_FUSED_GEMM_SPLIT_K", None)
    try:
        yield
    finally:
        if prior is not None:
            os.environ["FASTINFERENCE_AWQ_FUSED_GEMM_SPLIT_K"] = prior
        else:
            os.environ.pop("FASTINFERENCE_AWQ_FUSED_GEMM_SPLIT_K", None)


class TestEnvironmentOverrides:
    def test_env_integer_forces_value(self) -> None:
        with mock.patch.dict(os.environ, {"FASTINFERENCE_AWQ_FUSED_GEMM_SPLIT_K": "3"}):
            assert _select_split_k(1, 4096, 4096) == 3

    def test_env_clamps_below_one(self) -> None:
        with mock.patch.dict(os.environ, {"FASTINFERENCE_AWQ_FUSED_GEMM_SPLIT_K": "0"}):
            assert _select_split_k(1, 65536, 65536) == 1

    def test_env_rejects_garbage_and_falls_to_one(self) -> None:
        with mock.patch.dict(
            os.environ, {"FASTINFERENCE_AWQ_FUSED_GEMM_SPLIT_K": "banana"}
        ):
            assert _select_split_k(1, 65536, 65536) == 1


class TestAutoHeuristicDecode:
    """Decode (M=1) shape families. These are the hot path for TPS."""

    def test_gemma4_gate_up_fused_stays_split1(self) -> None:
        # gate_up pair: K=hidden_size~5376, N=2*intermediate~43008.
        # Shallow K means SPLIT_K=1 gives full occupancy via large N grid.
        assert _select_split_k(1, 43008, 5376) == 1

    def test_gemma4_down_proj_triggers_split4_narrow_n(self) -> None:
        # down_proj on Gemma4-31B: deep K, narrow N. This is the case we
        # added in Step 4.3; previously fell through to split_k=1.
        assert _select_split_k(1, 5376, 21504) == 4

    def test_qwen35_wide_deep_triggers_split4(self) -> None:
        # Qwen3.5-9B-class wide-output deep-K (pre-existing rule).
        assert _select_split_k(1, 8192, 16384) == 4

    def test_qwen35_down_proj_stays_split1(self) -> None:
        # Qwen3.5-9B down_proj: K=18944, N=3584. Narrow-N threshold is
        # 20480 precisely so this shape remains on split_k=1 (atomic
        # overhead would otherwise dominate on shallower K).
        assert _select_split_k(1, 3584, 18944) == 1

    def test_small_n_below_narrow_threshold_stays_split1(self) -> None:
        # Intentionally below the 4096 narrow-N floor.
        assert _select_split_k(1, 2048, 32768) == 1

    def test_tiny_k_stays_split1(self) -> None:
        assert _select_split_k(1, 4096, 4096) == 1


class TestAutoHeuristicPrefill:
    """M>1 (prefill / chunked prefill) should never split-K: there are
    plenty of M-tiles already to saturate the grid."""

    def test_prefill_m_small_never_splits(self) -> None:
        # Even deep-K prefill keeps split_k=1 so atomic cost is avoided.
        assert _select_split_k(4, 8192, 16384) == 1
        assert _select_split_k(8, 5376, 21504) == 1

    def test_prefill_m_large_never_splits(self) -> None:
        assert _select_split_k(128, 8192, 16384) == 1
        assert _select_split_k(512, 43008, 5376) == 1


class TestBoundaryShapes:
    """Edge cases: zero / one-sized inputs should not raise."""

    def test_zero_n_stays_split1(self) -> None:
        assert _select_split_k(1, 0, 16384) == 1

    def test_zero_k_stays_split1(self) -> None:
        assert _select_split_k(1, 4096, 0) == 1

    def test_zero_m_stays_split1(self) -> None:
        # Caller shouldn't hand us M=0 in practice, but defense in depth.
        assert _select_split_k(0, 8192, 16384) == 1
