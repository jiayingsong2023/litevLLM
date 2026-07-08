# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.tools.gemma4_26b_profile import run_awq_audit


def _parse_awq_audit_log(log_text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in log_text.splitlines():
        if "qkv_fused_decode" in line:
            counts["qkv_fused_decode"] = counts.get("qkv_fused_decode", 0) + 1
        elif "qkv_separate_decode" in line:
            counts["qkv_separate_decode"] = counts.get("qkv_separate_decode", 0) + 1
        elif "moe_int4_decode_used" in line:
            counts["moe_int4_decode_used"] = counts.get("moe_int4_decode_used", 0) + 1
        elif "moe_int4_decode_fallback" in line:
            counts["moe_int4_decode_fallback"] = counts.get("moe_int4_decode_fallback", 0) + 1
    return counts


def test_parse_awq_audit_log() -> None:
    sample = (
        "event=qkv_fused_decode\n"
        "event=qkv_fused_decode\n"
        "event=qkv_separate_decode\n"
        "event=moe_int4_decode_used\n"
    )
    result = _parse_awq_audit_log(sample)
    assert result["qkv_fused_decode"] == 2
    assert result["qkv_separate_decode"] == 1
    assert result["moe_int4_decode_used"] == 1
    assert result.get("moe_int4_decode_fallback", 0) == 0


@pytest.mark.skipif(
    os.environ.get("RUN_GEMMA4_26B_PERF") != "1",
    reason="Set RUN_GEMMA4_26B_PERF=1 to run heavy model load tests",
)
@pytest.mark.skipif(
    not Path("models/gemma-4-26B-A4B-it-AWQ-4bit").exists(),
    reason="Gemma4-26B model not present",
)
def test_26b_fastpath_coverage(tmp_path: Path) -> None:
    log_path = run_awq_audit(tmp_path, max_new_tokens=4)
    log_text = log_path.read_text(encoding="utf-8")
    counts = _parse_awq_audit_log(log_text)
    # In default BS=1 decode, fused QKV and MoE int4 should dominate.
    # NOTE: this parser is a starting point only; authoritative coverage is
    # derived from the P0 harness artifacts, not from this brittle regex.
    assert counts.get("qkv_fused_decode", 0) > counts.get("qkv_separate_decode", 0)
    assert counts.get("moe_int4_decode_used", 0) > 0
