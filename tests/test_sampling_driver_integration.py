# SPDX-License-Identifier: Apache-2.0
import torch

from vllm.engine.request_state import RequestState
from vllm.engine.sampling_driver import SamplingDriver
from vllm.policies.base import GenerationPolicies
from vllm.sampling_params import SamplingParams


class FakeTokenizer:
    eos_token_id = 2


class NoOpOutputProcessor:
    """Policies backend that performs no output processing."""

    def apply_context_bias(
        self,
        logits: torch.Tensor,
        generated_ids: list[int],
        sampling_params: SamplingParams,
        bias_token_ids: list[int] | None,
        is_capital_question: bool,
    ) -> torch.Tensor:
        """Return logits unchanged."""
        return logits


def test_driver_uses_vectorized_path_by_default() -> None:
    policies = GenerationPolicies(backend=NoOpOutputProcessor())
    driver = SamplingDriver(FakeTokenizer(), None, policies)
    sp = SamplingParams(temperature=0.0, repetition_penalty=1.2)
    requests = [
        RequestState(
            request_id="r0",
            prompt="",
            input_ids=[1],
            sampling_params=sp,
            generated_ids=[1],
        ),
        RequestState(
            request_id="r1",
            prompt="",
            input_ids=[1],
            sampling_params=sp,
            generated_ids=[2],
        ),
    ]
    logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    tokens = driver.sample_batch_tokens(logits, requests)
    assert isinstance(tokens, list)
    assert len(tokens) == 2


def _request(
    request_id: str,
    sampling_params: SamplingParams,
    rng: torch.Generator | None = None,
    generated_ids: list[int] | None = None,
) -> RequestState:
    return RequestState(
        request_id=request_id,
        prompt="",
        input_ids=[1],
        sampling_params=sampling_params,
        rng=rng,
        generated_ids=generated_ids or [],
    )


def test_vectorized_and_legacy_sampling_parity(monkeypatch) -> None:
    """Default vectorized path must match the legacy row-wise path."""
    vocab_size = 32
    batch_size = 4
    torch.manual_seed(7)
    logits = torch.randn(batch_size, vocab_size)
    sp = SamplingParams(
        temperature=0.7,
        top_k=10,
        top_p=0.9,
        repetition_penalty=1.2,
        frequency_penalty=0.1,
        presence_penalty=0.05,
    )

    vectorized_requests: list[RequestState] = []
    legacy_requests: list[RequestState] = []
    for i in range(batch_size):
        seed = 100 + i
        generated_ids = [i % vocab_size, (i + 1) % vocab_size]
        vectorized_requests.append(
            _request(
                f"v{i}",
                sp,
                torch.Generator(device=logits.device).manual_seed(seed),
                generated_ids=generated_ids,
            )
        )
        legacy_requests.append(
            _request(
                f"l{i}",
                sp,
                torch.Generator(device=logits.device).manual_seed(seed),
                generated_ids=generated_ids,
            )
        )

    vectorized_driver = SamplingDriver(
        FakeTokenizer(), None, GenerationPolicies(backend=NoOpOutputProcessor())
    )

    legacy_driver = SamplingDriver(
        FakeTokenizer(),
        None,
        GenerationPolicies(backend=NoOpOutputProcessor()),
        use_legacy=True,
    )

    vectorized_tokens = vectorized_driver.sample_batch_tokens(
        logits.clone(), vectorized_requests
    )
    legacy_tokens = legacy_driver.sample_batch_tokens(logits.clone(), legacy_requests)

    assert vectorized_tokens == legacy_tokens


def test_sample_next_token_matches_sample_batch_tokens() -> None:
    """Single-row helper should agree with the batch entrypoint."""
    vocab_size = 16
    torch.manual_seed(3)
    logits_1d = torch.randn(vocab_size)
    logits_2d = logits_1d.unsqueeze(0)
    sp = SamplingParams(
        temperature=0.8,
        top_k=5,
        top_p=0.85,
        frequency_penalty=0.1,
    )
    seed = 42
    generated_ids = [2, 4]

    req_next = _request(
        "next",
        sp,
        torch.Generator(device=logits_1d.device).manual_seed(seed),
        generated_ids=generated_ids,
    )
    req_batch = _request(
        "batch",
        sp,
        torch.Generator(device=logits_1d.device).manual_seed(seed),
        generated_ids=generated_ids,
    )

    driver = SamplingDriver(
        FakeTokenizer(), None, GenerationPolicies(backend=NoOpOutputProcessor())
    )
    next_token = driver.sample_next_token(logits_1d.clone(), req_next)
    batch_tokens = driver.sample_batch_tokens(logits_2d.clone(), [req_batch])

    assert batch_tokens == [next_token]
