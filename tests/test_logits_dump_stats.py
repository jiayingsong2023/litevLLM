# SPDX-License-Identifier: Apache-2.0
"""Tests for tests/tools/qwen35_moe_packed_lite_logits_audit.py summarize_logits_last_1d."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]


def _load_audit_module():
    p = _ROOT / "tests" / "tools" / "qwen35_moe_packed_lite_logits_audit.py"
    spec = importlib.util.spec_from_file_location("qwen35_moe_packed_lite_logits_audit", p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_summarize_logits_argmax_and_top5() -> None:
    mod = _load_audit_module()
    x = torch.tensor([0.0, 2.0, 1.0, -1.0, 3.0])
    s = mod.summarize_logits_last_1d(x)
    assert s["numel"] == 5
    assert s["finite_count"] == 5
    assert s["argmax"] == 4
    assert s["top5_indices"][0] == 4
    assert abs(s["top5_values"][0] - 3.0) < 1e-5


def test_summarize_logits_nan_inf() -> None:
    mod = _load_audit_module()
    x = torch.tensor([float("nan"), float("inf"), 1.0])
    s = mod.summarize_logits_last_1d(x)
    assert s["nan_count"] == 1
    assert s["inf_count"] == 1
    assert s["finite_count"] == 1
    assert s["argmax"] == 2
