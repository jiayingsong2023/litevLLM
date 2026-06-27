# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from vllm.engine.request_state import RequestState
from vllm.sampling_params import SamplingParams


def test_request_state_has_expected_defaults():
    req = RequestState(
        request_id="r1",
        prompt="hello",
        input_ids=[1, 2, 3],
        sampling_params=SamplingParams(),
    )
    assert req.is_prefill
    assert not req.finished
    assert req.seq_len == 0
    assert req.generated_ids == []


def test_request_state_is_multimodal():
    text_only = RequestState(
        request_id="r1", prompt="hi", input_ids=[1], sampling_params=SamplingParams()
    )
    assert not text_only.is_multimodal

    mm = RequestState(
        request_id="r2",
        prompt="img",
        input_ids=[1],
        sampling_params=SamplingParams(),
        multi_modal_data={"image": [{"image": "url"}]},
    )
    assert mm.is_multimodal
