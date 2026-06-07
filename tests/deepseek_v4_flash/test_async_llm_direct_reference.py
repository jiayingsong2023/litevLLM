from __future__ import annotations

import torch

from vllm.engine.async_llm import AsyncLLM


class _FakeModel:
    def generate_greedy_reference(
        self,
        input_ids: torch.Tensor,
        *,
        max_tokens: int,
    ) -> torch.Tensor:
        assert input_ids.tolist() == [7]
        assert max_tokens == 1
        return torch.tensor([7, 42], dtype=torch.long)


class _FakeTokenizer:
    eos_token_id = 0

    def encode(self, prompt: str) -> list[int]:
        assert prompt == "hello"
        return [3, 7]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        assert ids == [42]
        assert skip_special_tokens is True
        return "world"


class _FakeEngine:
    model = _FakeModel()
    tokenizer = _FakeTokenizer()


def test_async_llm_direct_reference_chat_uses_model_hook() -> None:
    llm = AsyncLLM.__new__(AsyncLLM)
    llm.engine = _FakeEngine()

    text = llm.generate_greedy_reference_chat("hello", max_tokens=1)

    assert text == "world"
