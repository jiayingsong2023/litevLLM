# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

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


def test_run_target_logits_tie_word_embeddings(proto_mod: Any) -> None:
    vocab_size = 7
    hidden_size = 4
    seq_len = 5
    fake_weight = torch.randn(vocab_size, hidden_size)
    fake_hidden = torch.randn(1, seq_len, hidden_size)

    inner = type(
        "FakeInner",
        (),
        {
            "config": SimpleNamespace(
                tie_word_embeddings=True,
                final_logit_softcapping=30.0,
            ),
            "embed_tokens": SimpleNamespace(weight=fake_weight),
            "layers": [None, None],
            "__call__": lambda self, *args, **kwargs: fake_hidden,
            "parameters": lambda self: iter([fake_weight]),
        },
    )()
    llm = SimpleNamespace(
        model=SimpleNamespace(model=inner),
        engine=SimpleNamespace(inf_config={"dummy": True}),
    )

    logits = proto_mod.run_target_logits(llm, torch.tensor([1, 2, 3, 4, 5]))

    assert logits.shape == (1, seq_len, vocab_size)
    assert logits.abs().max().item() <= 30.5


def test_run_target_logits_untied_lm_head(proto_mod: Any) -> None:
    vocab_size = 7
    hidden_size = 4
    seq_len = 3
    fake_hidden = torch.randn(1, seq_len, hidden_size)
    lm_logits = torch.randn(1, seq_len, vocab_size)

    lm_head = type(
        "FakeLMHead",
        (),
        {"__call__": lambda self, hidden, lora_mapping: lm_logits},
    )()
    inner = type(
        "FakeInner",
        (),
        {
            "config": SimpleNamespace(
                tie_word_embeddings=False,
                final_logit_softcapping=None,
            ),
            "embed_tokens": SimpleNamespace(
                weight=torch.randn(vocab_size, hidden_size)
            ),
            "layers": [None],
            "__call__": lambda self, *args, **kwargs: fake_hidden,
            "parameters": lambda self: iter([torch.randn(1)]),
        },
    )()
    llm = SimpleNamespace(
        model=SimpleNamespace(model=inner, lm_head=lm_head),
        engine=SimpleNamespace(inf_config={"dummy": True}),
    )

    logits = proto_mod.run_target_logits(llm, torch.tensor([1, 2, 3]))

    assert logits.shape == (1, seq_len, vocab_size)
    assert torch.equal(logits, lm_logits)
