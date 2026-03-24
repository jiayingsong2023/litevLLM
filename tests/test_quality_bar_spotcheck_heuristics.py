# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Tier-B substance heuristics in tests/tools/quality_bar_spotcheck.py."""
from __future__ import annotations

import importlib.util
import json
import os
import tempfile
from pathlib import Path
from typing import Any, List

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load_spotcheck():
    p = _ROOT / "tests" / "tools" / "quality_bar_spotcheck.py"
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


def test_coherence_fails_on_cjk_char_dominance_spam(qb: Any) -> None:
    """Repeated single CJK char mixed with junk (DeepSeek GGUF failure mode) must fail Tier-B."""
    spam = (
        "上丘座!1挺?花第二时间上海上留范上上上特别套上工作走上上雅工程那么上$上上Part-流行告别秋天，"
        "3上上上上子省2时代活的这么结束上告刚刚登上上整上挺,更多挺H一些"
    )
    ids = list(range(96))
    severe, detail, _ = qb.analyze_tier_b(spam, ids, _FakeTok(), check_substance=True)
    assert severe
    assert detail["coherence"]["pass"] is False
    assert detail["tier_b_alignment"]["coherence_ok"] is False


def test_coherence_ok_for_natural_english_letter_skew(qb: Any) -> None:
    """Latin prose can have one letter at ~14–16% (e.g. 'e'); must not fail dominance alone."""
    text = (
        "A binary search tree (BST) is a type of binary tree where each node has at most two children, "
        "referred to as the left and right child. In a BST, for any given node, the value of all nodes "
        "in the left subtree is less than the value of the node, and the value of all nodes in the right "
        "subtree is greater than the value of the node."
    )
    ids = list(range(96))
    severe, detail, _ = qb.analyze_tier_b(text, ids, _FakeTok(), check_substance=True)
    assert detail["coherence"]["pass"] is True
    assert detail["tier_b_alignment"]["coherence_ok"] is True


def test_coherence_fails_on_low_token_diversity(qb: Any) -> None:
    """Many tokens but few unique ids (loop) should fail."""
    text = "x " * 80 + "y"
    ids = [1] * 20 + [2] * 20 + [3] * 20 + [4] * 20 + [5] * 16
    severe, detail, _ = qb.analyze_tier_b(text, ids, _FakeTok(), check_substance=True)
    assert severe
    assert detail["coherence"]["pass"] is False


def test_model_config_hints_deepseek_moe_from_n_routed_experts(qb: Any) -> None:
    """DeepSeek V2 GGUF config uses n_routed_experts (not num_experts) — must enable MoE Tier-B defaults."""
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"model_type": "deepseek_v2", "n_routed_experts": 64}, f)
        h = qb._model_config_hints(d)
    assert h["model_type"] == "deepseek_v2"
    assert h["is_moe"] is True
