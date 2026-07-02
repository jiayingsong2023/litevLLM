# SPDX-License-Identifier: Apache-2.0
import torch

from vllm.engine.request_state import RequestState
from vllm.engine.sampling.sampler import Sampler
from vllm.engine.sampling_driver import SamplingDriver
from vllm.sampling_params import SamplingParams


class _NoOpPolicies:
    def apply_context_bias(
        self,
        logits: torch.Tensor,
        generated_ids: list[int],
        sampling_params: SamplingParams,
        bias_token_ids: list[int] | None,
        is_capital_question: bool,
    ) -> torch.Tensor:
        return logits


class _NoOpTokenizer:
    eos_token_id: int | None = None


def _reference_driver() -> SamplingDriver:
    return SamplingDriver(
        tokenizer=_NoOpTokenizer(),
        hf_config=None,
        policies=_NoOpPolicies(),
    )


def _request(
    request_id: str,
    sampling_params: SamplingParams,
    rng: torch.Generator | None = None,
) -> RequestState:
    return RequestState(
        request_id=request_id,
        prompt="",
        input_ids=[1],
        sampling_params=sampling_params,
        rng=rng,
    )


def test_sampler_greedy_batch() -> None:
    sp = SamplingParams(temperature=0.0)
    requests = [
        _request("r0", sp),
        _request("r1", sp),
    ]
    logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    sampler = Sampler()
    tokens = sampler.sample(logits, requests)
    assert tokens == [2, 0]


def test_sampler_non_greedy_shape_and_properties() -> None:
    sp = SamplingParams(temperature=0.7, top_k=2, top_p=0.9)
    requests = [
        _request("r0", sp),
        _request("r1", sp),
    ]
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    sampler = Sampler()
    tokens = sampler.sample(logits, requests)

    assert len(tokens) == len(requests)
    assert all(isinstance(t, int) for t in tokens)
    assert all(0 <= t < logits.shape[-1] for t in tokens)


def test_sampler_non_greedy_matches_reference_driver() -> None:
    torch.manual_seed(42)
    batch_size = 4
    vocab_size = 20
    logits = torch.randn(batch_size, vocab_size)
    sp = SamplingParams(temperature=0.7, top_k=5, top_p=0.9)

    requests_sampler: list[RequestState] = []
    requests_reference: list[RequestState] = []
    for i in range(batch_size):
        seed = 123 + i
        rng_sampler = torch.Generator(device=logits.device).manual_seed(seed)
        rng_reference = torch.Generator(device=logits.device).manual_seed(seed)
        requests_sampler.append(_request(f"r{i}", sp, rng=rng_sampler))
        requests_reference.append(_request(f"r{i}", sp, rng=rng_reference))

    sampler = Sampler()
    sampler_tokens = sampler.sample(logits.clone(), requests_sampler)

    reference_tokens = _reference_driver().sample_batch_tokens(
        logits.clone(), requests_reference
    )

    assert sampler_tokens == reference_tokens


def test_sampler_mixed_greedy_non_greedy() -> None:
    sp_greedy = SamplingParams(temperature=0.0)
    sp_non_greedy = SamplingParams(temperature=0.7, top_k=3, top_p=0.9)
    requests = [
        _request("r0", sp_greedy),
        _request("r1", sp_non_greedy),
        _request("r2", sp_greedy),
    ]
    logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [0.5, 1.5, 2.5]])
    sampler = Sampler()
    tokens = sampler.sample(logits, requests)

    assert len(tokens) == len(requests)
    assert tokens[0] == 2
    assert tokens[2] == 2
    assert 0 <= tokens[1] < logits.shape[-1]
