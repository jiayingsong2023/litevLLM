# SPDX-License-Identifier: Apache-2.0
"""Tests for verify_semantic_integrity prefill alignment helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _load_verify():
    p = _ROOT / "tests" / "verify_semantic_integrity.py"
    spec = importlib.util.spec_from_file_location("verify_semantic_integrity", p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_prefill_hf_alignment_pass() -> None:
    mod = _load_verify()
    assert mod.prefill_hf_alignment_pass(0.999, 42, 42) is True
    assert mod.prefill_hf_alignment_pass(0.99, 42, 42) is False
    assert mod.prefill_hf_alignment_pass(
        0.994, 42, 42, cos_min=mod.PREFILL_COSIM_MIN_GGUF_VS_FP16
    ) is True
    assert mod.prefill_hf_alignment_pass(
        0.993, 42, 42, cos_min=mod.PREFILL_COSIM_MIN_AWQ_VS_FP16
    ) is True
    assert mod.prefill_hf_alignment_pass(0.999, 1, 2) is False
    assert mod.prefill_hf_alignment_pass(None, 1, 1) is False


def test_prefill_cosine_floor_for_hf_compare() -> None:
    mod = _load_verify()
    f = mod.prefill_cosine_floor_for_hf_compare
    assert f("/tmp/same", None, "none") == mod.PREFILL_COSIM_MIN
    assert f("/tmp/same", "/tmp/same", "awq") == mod.PREFILL_COSIM_MIN
    assert f("/tmp/lite", "/tmp/hf_fp16", "awq") == mod.PREFILL_COSIM_MIN_AWQ_VS_FP16
    assert f("/tmp/lite", "/tmp/hf_fp16", "gguf") == mod.PREFILL_COSIM_MIN_GGUF_VS_FP16
    assert f("/tmp/lite", "/tmp/hf_fp16", "none") == mod.PREFILL_COSIM_MIN
