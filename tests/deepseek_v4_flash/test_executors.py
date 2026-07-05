from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm.engine.executor_result import TokenDecodeResult, TokenPrefillResult
from vllm.engine.request_state import RequestState
from vllm.model_executor.models.deepseek_v4_flash.executors import (
    DeepSeekDecodeExecutor,
    DeepSeekPrefillExecutor,
)
from vllm.model_executor.models.deepseek_v4_flash.model import (
    DeepSeekV4FlashForCausalLM,
)
from vllm.sampling_params import SamplingParams


class _Scheduler:
    def __init__(self, request: RequestState) -> None:
        self.request = request

    def get_request(self, request_id: str) -> RequestState:
        assert request_id == self.request.request_id
        return self.request


class _Model:
    def __init__(self) -> None:
        self.prefill_calls: list[tuple[str, list[int], int]] = []
        self.decode_calls: list[str] = []

    def device(self) -> torch.device:
        return torch.device("cpu")

    def prefill_request(
        self,
        request_id: str,
        input_ids: list[int],
        max_tokens: int,
    ) -> int:
        self.prefill_calls.append((request_id, input_ids, max_tokens))
        return 42

    def decode_single_token(self, request_id: str) -> int:
        self.decode_calls.append(request_id)
        return 43


def test_deepseek_prefill_executor_returns_token_result() -> None:
    request = RequestState(
        request_id="req-1",
        prompt="hello",
        input_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=5, temperature=0.0),
    )
    model = _Model()

    result = DeepSeekPrefillExecutor(model=model, observer=None).execute(
        ["req-1"],
        _Scheduler(request),
        chunk_len=3,
    )

    assert isinstance(result, TokenPrefillResult)
    assert result.prefilled_tokens == [3]
    assert result.is_last_chunk == [True]
    torch.testing.assert_close(result.next_token_ids, torch.tensor([42]))
    assert model.prefill_calls == [("req-1", [1, 2, 3], 5)]


def test_deepseek_decode_executor_returns_token_result() -> None:
    request = RequestState(
        request_id="req-1",
        prompt="hello",
        input_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=5, temperature=0.0),
    )
    model = _Model()

    result = DeepSeekDecodeExecutor(model=model, observer=None).execute_batch(
        ["req-1"],
        _Scheduler(request),
    )

    assert isinstance(result, TokenDecodeResult)
    torch.testing.assert_close(result.next_token_ids, torch.tensor([43]))
    assert model.decode_calls == ["req-1"]


def test_model_prefill_request_stores_session_under_external_request_id(
    monkeypatch,
) -> None:
    model = DeepSeekV4FlashForCausalLM()
    session = SimpleNamespace(state=object())
    calls: list[tuple[list[int], int, str | None]] = []

    def fake_prefill(input_ids, *, max_tokens, request_id=None):
        calls.append((input_ids.tolist(), max_tokens, request_id))
        return session

    monkeypatch.setattr(model, "prefill_greedy_kernel", fake_prefill)
    monkeypatch.setattr(model, "decode_single_token", lambda request_id: 99)

    token = model.prefill_request("req-1", [1, 2, 3], 5)

    assert token == 99
    assert calls == [([1, 2, 3], 5, "req-1")]
    assert model._gpu_sessions["req-1"] is session
