# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Tier-B substance heuristics in scripts/quality_bar_spotcheck.py."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, List

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load_spotcheck():
    p = _ROOT / "scripts" / "quality_bar_spotcheck.py"
    spec = importlib.util.spec_from_file_location("quality_bar_spotcheck", p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def qb() -> Any:
    return _load_spotcheck()


class _FakeTok:
    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        if not token_ids:
            return ""
        return {198: "\n", 72: "T", 100: "The"}.get(token_ids[0], "x")


def test_nonspace_helper(qb: Any) -> None:
    assert qb._nonspace_chars(" a \n b ") == "ab"


def test_substance_whitespace_only_fails(qb: Any) -> None:
    severe, detail, _ = qb.analyze_tier_b("\n\n", [198, 198], _FakeTok(), check_substance=True)
    assert severe
    assert detail["substance"]["pass"] is False
    assert detail["tier_b_alignment"]["substance_ok"] is False


def test_substance_ok_for_normal_sentence(qb: Any) -> None:
    text = "The capital of France is Paris."
    ids = list(range(12))
    severe, detail, _ = qb.analyze_tier_b(text, ids, _FakeTok(), check_substance=True)
    assert detail["substance"]["pass"] is True
    assert detail["tier_b_alignment"]["substance_ok"] is True


def test_substance_digit_heavy_fails(qb: Any) -> None:
    text = "1.3\n5 0\n\n30 4 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1"
    ids = list(range(40))
    severe, detail, msgs = qb.analyze_tier_b(text, ids, _FakeTok(), check_substance=True)
    assert severe
    assert detail["substance"]["pass"] is False
    assert any("digit" in m or "letter" in m for m in msgs)


def test_check_substance_disabled(qb: Any) -> None:
    severe, detail, _ = qb.analyze_tier_b("\n\n", [1, 2], _FakeTok(), check_substance=False)
    assert detail["substance"]["pass"] is True


class _FragTok:
    """First id decodes to lone U+FFFD; prefix decode is valid (BPE fragment case)."""

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        if not token_ids:
            return ""
        if token_ids == [99]:
            return "\ufffd"
        if token_ids[:4] == [99, 100, 101, 102]:
            return "hello world"
        return "x"


def test_first_token_fragment_not_severe_when_full_text_ok(qb: Any) -> None:
    text = "hello world"
    ids = [99, 100, 101, 102]
    severe, detail, _ = qb.analyze_tier_b(text, ids, _FragTok(), check_substance=True)
    assert detail["first_token"]["pass"] is True
    assert detail["tier_b_alignment"]["first_token_ok"] is True
