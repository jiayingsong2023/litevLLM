# SPDX-License-Identifier: Apache-2.0
import torch

from vllm.engine.request_state import RequestState
from vllm.engine.sampling_driver import SamplingDriver
from vllm.policies.base import GenerationPolicies
from vllm.sampling_params import SamplingParams


class FakeTokenizer:
    eos_token_id = 2


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
