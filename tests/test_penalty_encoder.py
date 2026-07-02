# SPDX-License-Identifier: Apache-2.0
import torch

from vllm.engine.request_state import RequestState
from vllm.engine.sampling.penalty_encoder import PenaltyEncoder
from vllm.policies.base import GenerationPolicies
from vllm.sampling_params import SamplingParams


class NoOpOutputProcessor:
    def apply_context_bias(
        self,
        logits,
        generated_ids,
        sampling_params,
        bias_token_ids,
        is_capital_question,
    ):
        return logits


def _make_driver():
    class FakeTokenizer:
        eos_token_id = 2

    policies = GenerationPolicies(backend=NoOpOutputProcessor())
    return PenaltyEncoder(FakeTokenizer(), None, policies)


def _assert_parity(encoder, logits, requests):
    vectorized = encoder.encode(logits, requests)
    for i, req in enumerate(requests):
        expected = encoder.encode_row(logits[i], req)
        assert torch.allclose(vectorized[i], expected, atol=1e-5)


def test_vectorized_matches_per_row_repetition_penalty() -> None:
    encoder = _make_driver()
    vocab_size = 128
    requests = []
    for i in range(4):
        sp = SamplingParams(temperature=0.7, repetition_penalty=1.0 + i * 0.1)
        req = RequestState(
            request_id=f"r{i}",
            prompt="",
            input_ids=[1, 2, 3],
            sampling_params=sp,
            generated_ids=[10, 20, 30, 10],
        )
        requests.append(req)
    logits = torch.randn(4, vocab_size)
    _assert_parity(encoder, logits, requests)


def test_vectorized_matches_per_row_frequency_penalty() -> None:
    encoder = _make_driver()
    vocab_size = 128
    requests = []
    for i in range(4):
        sp = SamplingParams(temperature=0.7, frequency_penalty=0.1 * (i + 1))
        req = RequestState(
            request_id=f"r{i}",
            prompt="",
            input_ids=[1, 2, 3],
            sampling_params=sp,
            generated_ids=[5, 5, 6, 7, 7, 7],
        )
        requests.append(req)
    logits = torch.randn(4, vocab_size)
    _assert_parity(encoder, logits, requests)


def test_vectorized_matches_per_row_presence_penalty() -> None:
    encoder = _make_driver()
    vocab_size = 128
    requests = []
    for i in range(4):
        sp = SamplingParams(temperature=0.7, presence_penalty=0.2 * (i + 1))
        req = RequestState(
            request_id=f"r{i}",
            prompt="",
            input_ids=[1, 2, 3],
            sampling_params=sp,
            generated_ids=[8, 9, 8, 10],
        )
        requests.append(req)
    logits = torch.randn(4, vocab_size)
    _assert_parity(encoder, logits, requests)


def test_vectorized_matches_per_row_eos_min_tokens() -> None:
    encoder = _make_driver()
    vocab_size = 128
    requests = []
    for min_tokens in [0, 3, 5]:
        sp = SamplingParams(temperature=0.7, min_tokens=min_tokens)
        req = RequestState(
            request_id=f"min{min_tokens}",
            prompt="",
            input_ids=[1, 2, 3],
            sampling_params=sp,
            generated_ids=[1] * min(2, max(min_tokens, 1)),
        )
        requests.append(req)
    logits = torch.randn(3, vocab_size)
    _assert_parity(encoder, logits, requests)


def test_vectorized_matches_per_row_ignore_eos() -> None:
    encoder = _make_driver()
    vocab_size = 128
    sp = SamplingParams(temperature=0.7, ignore_eos=True)
    req = RequestState(
        request_id="ignore",
        prompt="",
        input_ids=[1, 2, 3],
        sampling_params=sp,
        generated_ids=[1, 2, 3],
    )
    logits = torch.randn(1, vocab_size)
    _assert_parity(encoder, logits, [req])


def test_vectorized_matches_per_row_anti_template() -> None:
    encoder = _make_driver()
    vocab_size = 128
    requests = []
    for i in range(3):
        sp = SamplingParams(temperature=0.7)
        req = RequestState(
            request_id=f"anti{i}",
            prompt="",
            input_ids=[1, 2, 3],
            sampling_params=sp,
            generated_ids=[1] * i,
            anti_template_token_ids=[11, 12, 13],
        )
        requests.append(req)
    logits = torch.randn(3, vocab_size)
    _assert_parity(encoder, logits, requests)


def test_mixed_batch_with_fallback_rows() -> None:
    encoder = _make_driver()
    vocab_size = 128

    requests = []
    for i in range(4):
        sp = SamplingParams(temperature=0.7, repetition_penalty=1.0 + i * 0.1)
        req = RequestState(
            request_id=f"r{i}",
            prompt="",
            input_ids=[1, 2, 3],
            sampling_params=sp,
            generated_ids=[10, 20, 30, 10],
        )
        if i == 1:
            req.capital_question_bias_token_ids = [7]
        if i == 3:
            req.is_chinese_capital_question = True
        requests.append(req)

    logits = torch.randn(4, vocab_size)
    _assert_parity(encoder, logits, requests)


def test_vectorized_matches_per_row_out_of_range_generated_ids() -> None:
    encoder = _make_driver()
    vocab_size = 128
    requests = []
    for i in range(4):
        sp = SamplingParams(
            temperature=0.7,
            repetition_penalty=1.1,
            frequency_penalty=0.1,
            presence_penalty=0.2,
        )
        req = RequestState(
            request_id=f"oor{i}",
            prompt="",
            input_ids=[1, 2, 3],
            sampling_params=sp,
            generated_ids=[-10, -1, 10, 20, 30, 127, 128, 200],
        )
        requests.append(req)
    logits = torch.randn(4, vocab_size)
    _assert_parity(encoder, logits, requests)
