# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
import torch

from vllm.engine.sampling_driver import SamplingDriver
from vllm.sampling_params import SamplingParams


class DummyTokenizer:
    def __init__(self):
        self.eos_token_id = 99


class DummyPolicies:
    def apply_context_bias(
        self,
        logits: torch.Tensor,
        generated_ids: list[int],
        sampling_params: Any,
        bias_token_ids: list[int] | None = None,
        is_capital_question: bool = False,
    ) -> torch.Tensor:
        return logits


def test_2d_batch_sampler_greedy():
    tokenizer = DummyTokenizer()
    policies = DummyPolicies()
    driver = SamplingDriver(tokenizer, None, policies)

    # Batch of size 3, Vocab size 5
    # All are greedy requests
    logits = torch.tensor(
        [
            [1.0, 5.0, 2.0, 0.0, -1.0],  # argmax is 1
            [9.0, 0.0, 1.0, 2.0, 3.0],  # argmax is 0
            [-5.0, -1.0, -2.0, -10.0, 0.0],  # argmax is 4
        ],
        dtype=torch.float32,
    )

    requests = [
        {
            "generated_ids": [],
            "sampling_params": SamplingParams(temperature=0.0),
        },
        {
            "generated_ids": [1, 2],
            "sampling_params": SamplingParams(temperature=0.0),
        },
        {
            "generated_ids": [],
            "sampling_params": SamplingParams(temperature=0.0),
        },
    ]

    tokens = driver.sample_batch_tokens(logits, requests)
    assert tokens == [1, 0, 4]


def test_2d_batch_sampler_greedy_skips_sort_and_softmax(
    monkeypatch: pytest.MonkeyPatch,
):
    tokenizer = DummyTokenizer()
    policies = DummyPolicies()
    driver = SamplingDriver(tokenizer, None, policies)

    logits = torch.tensor(
        [
            [1.0, 5.0, 2.0],
            [9.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    requests = [
        {"generated_ids": [], "sampling_params": SamplingParams(temperature=0.0)},
        {"generated_ids": [], "sampling_params": SamplingParams(temperature=0.0)},
    ]

    def _unexpected_sort(*args, **kwargs):
        raise AssertionError("greedy sampling should not sort the vocab")

    def _unexpected_softmax(*args, **kwargs):
        raise AssertionError("greedy sampling should not compute softmax")

    monkeypatch.setattr(torch, "sort", _unexpected_sort)
    monkeypatch.setattr(torch, "softmax", _unexpected_softmax)

    assert driver.sample_batch_tokens(logits, requests) == [1, 0]


def test_2d_batch_sampler_mixed_params_and_penalties():
    tokenizer = DummyTokenizer()
    policies = DummyPolicies()
    driver = SamplingDriver(tokenizer, None, policies)

    # Batch of size 4, Vocab size 5
    logits = torch.tensor(
        [
            # Req 0: greedy, no penalty -> argmax is 2
            [0.0, 1.0, 10.0, 2.0, 0.0],
            # Req 1: greedy, with repetition penalty on token 2 (rp = 5.0)
            # scaled down to 2.0 -> argmax is 3
            [0.0, 1.0, 10.0, 3.0, 0.0],
            # Req 2: greedy, with presence penalty on token 2 (pp = 9.0)
            # reduced to 1.0 -> argmax is 3
            [0.0, 1.0, 10.0, 3.0, 0.0],
            # Req 3: greedy, with frequency penalty on token 2
            # (fp = 4.0, count = 2) -> reduced to 2.0 -> argmax is 3
            [0.0, 1.0, 10.0, 3.0, 0.0],
        ],
        dtype=torch.float32,
    )

    requests = [
        {
            "generated_ids": [],
            "sampling_params": SamplingParams(temperature=0.0),
        },
        {
            "generated_ids": [2],
            "sampling_params": SamplingParams(temperature=0.0, repetition_penalty=5.0),
        },
        {
            "generated_ids": [2],
            "sampling_params": SamplingParams(temperature=0.0, presence_penalty=9.0),
        },
        {
            "generated_ids": [2, 2],
            "sampling_params": SamplingParams(temperature=0.0, frequency_penalty=4.0),
        },
    ]

    tokens = driver.sample_batch_tokens(logits, requests)
    assert tokens == [2, 3, 3, 3]


def test_2d_batch_sampler_top_k_top_p():
    tokenizer = DummyTokenizer()
    policies = DummyPolicies()
    driver = SamplingDriver(tokenizer, None, policies)

    # 10 tokens vocabulary, batch of size 2
    logits = torch.tensor(
        [
            # Req 0: non-greedy (temp=1.0), top_k=3 -> only tokens 0, 1, 2
            # should have non-inf logits
            [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            # Req 1: non-greedy (temp=1.0), top_p=0.4 -> only top tokens
            # making up 0.4 cumulative prob should be kept.
            # Logits are [10.0, 9.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
            # Softmaxed probs for top 2 will be extremely close to 1.0,
            # so only token 0 and 1 are kept.
            [10.0, 9.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        ],
        dtype=torch.float32,
    )

    requests = [
        {
            "generated_ids": [],
            "sampling_params": SamplingParams(temperature=1.0, top_k=3),
        },
        {
            "generated_ids": [],
            "sampling_params": SamplingParams(temperature=1.0, top_p=0.4),
        },
    ]

    # Run multiple times to verify token distribution is constrained
    for _ in range(50):
        tokens = driver.sample_batch_tokens(logits, requests)
        assert tokens[0] in [0, 1, 2]
        assert tokens[1] in [0, 1]


def test_2d_batch_sampler_empty_and_shapes():
    tokenizer = DummyTokenizer()
    policies = DummyPolicies()
    driver = SamplingDriver(tokenizer, None, policies)

    # Empty requests list
    assert driver.sample_batch_tokens(torch.empty(0), []) == []

    # 1D single-token squeeze test
    logits_1d = torch.tensor([1.0, 5.0, 2.0], dtype=torch.float32)
    requests = [
        {"generated_ids": [], "sampling_params": SamplingParams(temperature=0.0)}
    ]
    assert driver.sample_batch_tokens(logits_1d, requests) == [1]

    # 3D squeeze test
    logits_3d = torch.tensor([[[1.0, 2.0, 10.0]]], dtype=torch.float32)
    assert driver.sample_batch_tokens(logits_3d, requests) == [2]
