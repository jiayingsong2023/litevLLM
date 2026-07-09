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


def _make_mock_llm(reference_generated: list[int]) -> Any:
    """Return a mock LLM whose generate() emits reference_generated."""
    completion = SimpleNamespace(token_ids=list(reference_generated), text="")
    output = SimpleNamespace(
        prompt_token_ids=[1, 2, 3],
        outputs=[completion],
    )

    def _generate(prompts: list[str], sampling_params: Any) -> list[Any]:
        return [output]

    return SimpleNamespace(generate=_generate)


def _mock_run_target_logits(reference: list[int], prompt_len: int):
    """Return logits whose argmax matches the reference at each generated position.

    Sets the target logit at index p-1 for every input position p in
    [prompt_len, L], using reference[p-prompt_len] when within range, so the
    bonus token past the last proposed draft is also defined.
    """

    def _run(llm: Any, input_ids: torch.Tensor) -> torch.Tensor:
        ids = input_ids[0].tolist()
        vocab_size = max(reference + ids) + 10
        logits = torch.full((1, len(ids), vocab_size), -1e9)
        for p in range(prompt_len, len(ids) + 1):
            ref_idx = p - prompt_len
            if ref_idx < len(reference):
                logits[0, p - 1, reference[ref_idx]] = 1e6
        return logits

    return _run


def test_speculative_decode_bit_exact_when_drafts_match(
    proto_mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = [10, 11, 12, 13]
    prompt = [1, 2, 3]
    mock_llm = _make_mock_llm(reference)
    monkeypatch.setattr(
        proto_mod, "run_target_logits", _mock_run_target_logits(reference, len(prompt))
    )

    def draft_proposer(prefix: list[int], generated: list[int], k: int) -> list[int]:
        # Perfect oracle: return the next reference tokens.
        start = len(generated)
        return reference[start : start + k]

    result = proto_mod.speculative_decode(
        mock_llm,
        draft_proposer,
        prompt_token_ids=prompt,
        max_new_tokens=len(reference),
        num_draft_tokens=2,
    )

    assert result["token_ids"] == reference
    assert result["acceptance_rate"] == 1.0
    assert result["accepted_total"] <= len(reference)


def test_speculative_decode_recovers_on_mismatch(
    proto_mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = [10, 11, 12]
    prompt = [1, 2, 3]
    mock_llm = _make_mock_llm(reference)
    monkeypatch.setattr(
        proto_mod, "run_target_logits", _mock_run_target_logits(reference, len(prompt))
    )

    calls: list[int] = []

    def draft_proposer(prefix: list[int], generated: list[int], k: int) -> list[int]:
        calls.append(len(generated))
        if len(generated) == 0:
            return [99, reference[1]]  # first draft token mismatches
        return reference[len(generated) : len(generated) + k]

    result = proto_mod.speculative_decode(
        mock_llm,
        draft_proposer,
        prompt_token_ids=prompt,
        max_new_tokens=len(reference),
        num_draft_tokens=2,
    )

    assert result["token_ids"] == reference
    assert result["accepted_total"] < result["proposed_total"]
