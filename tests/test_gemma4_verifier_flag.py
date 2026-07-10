from types import SimpleNamespace
from typing import Any

import pytest  # noqa: F401
import torch
import torch.nn as nn

from vllm.model_executor.models.gemma4.model import Gemma4ForConditionalGeneration


class FakeLMHead:
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        self.weight = torch.randn(vocab_size, hidden_size)

    def __call__(self, hidden: torch.Tensor, lora_mapping: Any) -> torch.Tensor:
        return torch.nn.functional.linear(hidden, self.weight)


def _make_model(hidden_size: int, vocab_size: int, tie: bool) -> Any:
    model = object.__new__(Gemma4ForConditionalGeneration)
    nn.Module.__init__(model)
    inner = torch.nn.Module()
    inner.config = SimpleNamespace(tie_word_embeddings=tie)
    inner.forward = lambda *args, **kwargs: torch.randn(1, 5, hidden_size)
    model.model = inner
    if tie:
        inner.embed_tokens = SimpleNamespace(
            weight=torch.randn(vocab_size, hidden_size)
        )
        model.lm_head = None
    else:
        model.lm_head = FakeLMHead(vocab_size, hidden_size)
    return model


def test_verifier_flag_returns_all_logits():
    model = _make_model(hidden_size=16, vocab_size=8, tie=False)
    attn_metadata = {"verifier_return_all_logits": True}
    logits = Gemma4ForConditionalGeneration.forward(
        model,
        torch.tensor([[1, 2, 3, 4, 5]]),
        torch.tensor([[0, 1, 2, 3, 4]]),
        [],
        attn_metadata,
    )
    assert tuple(logits.shape) == (1, 5, 8)


def test_verifier_flag_false_returns_last_logits():
    model = _make_model(hidden_size=16, vocab_size=8, tie=False)
    attn_metadata = {"verifier_return_all_logits": False}
    logits = Gemma4ForConditionalGeneration.forward(
        model,
        torch.tensor([[1, 2, 3, 4, 5]]),
        torch.tensor([[0, 1, 2, 3, 4]]),
        [],
        attn_metadata,
    )
    assert tuple(logits.shape) == (1, 1, 8)


def test_verifier_flag_tied_embeddings():
    model = _make_model(hidden_size=16, vocab_size=8, tie=True)
    attn_metadata = {"verifier_return_all_logits": True}
    logits = Gemma4ForConditionalGeneration.forward(
        model,
        torch.tensor([[1, 2, 3, 4, 5]]),
        torch.tensor([[0, 1, 2, 3, 4]]),
        [],
        attn_metadata,
    )
    assert tuple(logits.shape) == (1, 5, 8)
