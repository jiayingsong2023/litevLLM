# SPDX-License-Identifier: Apache-2.0
import torch

from vllm.engine.request_state import RequestState
from vllm.engine.sampling.sampler import Sampler
from vllm.sampling_params import SamplingParams


def test_sampler_greedy_batch() -> None:
    sp = SamplingParams(temperature=0.0)
    requests = [
        RequestState(request_id="r0", prompt="", input_ids=[1], sampling_params=sp),
        RequestState(request_id="r1", prompt="", input_ids=[1], sampling_params=sp),
    ]
    logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    sampler = Sampler()
    tokens = sampler.sample(logits, requests)
    assert tokens == [2, 0]
