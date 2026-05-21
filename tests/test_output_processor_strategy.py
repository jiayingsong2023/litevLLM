# SPDX-License-Identifier: Apache-2.0

import torch

from vllm.entrypoints.output_processors import (
    DefaultOutputProcessor,
    OutputProcessorStrategy,
    VerlOutputProcessor,
)
from vllm.outputs import CompletionOutput, RequestOutput


def test_default_output_processor():
    processor = DefaultOutputProcessor()
    assert isinstance(processor, OutputProcessorStrategy)

    comp1 = CompletionOutput(
        index=0, text="hello", token_ids=[1, 2, 3], cumulative_logprob=-0.5
    )
    req1 = RequestOutput(
        request_id="req1",
        prompt="hi",
        prompt_token_ids=[1],
        outputs=[comp1],
        finished=True,
    )

    comp2 = CompletionOutput(
        index=0, text="world", token_ids=[4, 5], cumulative_logprob=-1.2
    )
    req2 = RequestOutput(
        request_id="req2",
        prompt="yo",
        prompt_token_ids=[2],
        outputs=[comp2],
        finished=True,
    )

    outputs = [req1, req2]
    processed = processor.process_outputs(outputs)

    assert processed == outputs
    assert len(processed) == 2
    assert processed[0].outputs[0].text == "hello"
    assert processed[1].outputs[0].text == "world"


def test_verl_output_processor_padding():
    processor = VerlOutputProcessor()
    assert isinstance(processor, OutputProcessorStrategy)

    comp1 = CompletionOutput(
        index=0, text="hello", token_ids=[1, 2, 3], cumulative_logprob=-0.5
    )
    req1 = RequestOutput(
        request_id="req1",
        prompt="hi",
        prompt_token_ids=[1],
        outputs=[comp1],
        finished=True,
    )

    comp2 = CompletionOutput(
        index=0, text="world", token_ids=[4, 5], cumulative_logprob=-1.2
    )
    req2 = RequestOutput(
        request_id="req2",
        prompt="yo",
        prompt_token_ids=[2],
        outputs=[comp2],
        finished=True,
    )

    outputs = [req1, req2]

    # Test with default pad_token_id=0
    processed_default = processor.process_outputs(outputs)
    assert "token_ids" in processed_default
    assert "cumulative_logprob" in processed_default

    token_ids_tensor = processed_default["token_ids"]
    cum_logprob_tensor = processed_default["cumulative_logprob"]

    assert token_ids_tensor.shape == (2, 3)
    assert torch.equal(token_ids_tensor[0], torch.tensor([1, 2, 3], dtype=torch.long))
    assert torch.equal(token_ids_tensor[1], torch.tensor([4, 5, 0], dtype=torch.long))
    assert torch.allclose(
        cum_logprob_tensor, torch.tensor([-0.5, -1.2], dtype=torch.float32)
    )

    # Test with custom pad_token_id=-1
    processed_custom = processor.process_outputs(outputs, pad_token_id=-1)
    token_ids_tensor_custom = processed_custom["token_ids"]
    assert torch.equal(
        token_ids_tensor_custom[1],
        torch.tensor([4, 5, -1], dtype=torch.long),
    )


def test_verl_output_processor_empty():
    processor = VerlOutputProcessor()
    processed = processor.process_outputs([])
    assert processed == {}
