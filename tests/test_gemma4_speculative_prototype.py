# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load_module() -> Any:
    p = _ROOT / "tests" / "tools" / "gemma4_speculative_prototype.py"
    spec = importlib.util.spec_from_file_location("gemma4_speculative_prototype", p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def proto_mod() -> Any:
    return _load_module()


def test_propose_ngram_finds_repeated_suffix(proto_mod: Any) -> None:
    prefix = [1, 2, 3, 4, 5]
    generated = [2, 3]
    # The suffix [2, 3] appeared earlier starting at index 1 in
    # [1,2,3,4,5,2,3]. The 3 tokens after that earlier occurrence are [4, 5, 2].
    result = proto_mod.propose_ngram(prefix, generated, k=3, ngram_min=2, ngram_max=4)
    assert result == [4, 5, 2]


def test_propose_ngram_prefers_longer_needle(proto_mod: Any) -> None:
    prefix = [10, 20, 30, 40]
    generated = [10, 20, 30]
    # n=3 needle [10,20,30] matches at index 0; return [40, 10].
    result = proto_mod.propose_ngram(prefix, generated, k=2, ngram_min=2, ngram_max=3)
    assert result == [40, 10]


def test_propose_ngram_returns_empty_when_no_match(proto_mod: Any) -> None:
    result = proto_mod.propose_ngram([1, 2, 3], [4, 5], k=5, ngram_min=2, ngram_max=4)
    assert result == []


def test_propose_ngram_does_not_match_final_suffix(proto_mod: Any) -> None:
    prefix = [1, 2]
    generated = [3, 4]
    # The only [3,4] is the final suffix itself; no earlier occurrence exists.
    result = proto_mod.propose_ngram(prefix, generated, k=2, ngram_min=2, ngram_max=2)
    assert result == []
