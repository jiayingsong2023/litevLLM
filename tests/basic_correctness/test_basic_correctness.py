# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the short outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/basic_correctness/test_basic_correctness.py`.
"""

import pytest
import weakref
from vllm import LLM
from ..conftest import VllmRunner, HfRunner
from ..models.utils import check_outputs_equal

MODELS = [
    "hmellor/tiny-random-LlamaForCausalLM",
    "meta-llama/Llama-3.2-1B-Instruct",
]

def test_vllm_gc_ed():
    """Verify vllm instance is GC'ed when it is deleted"""
    llm = LLM("hmellor/tiny-random-LlamaForCausalLM")
    weak_llm = weakref.ref(llm)
    del llm
    assert weak_llm() is None

@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [5])
def test_models(
    hf_runner,
    model: str,
    max_tokens: int,
) -> None:
    example_prompts = ["The capital of France is"]

    with hf_runner(model) as hf_model:
        hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)

    with VllmRunner(
        model,
        max_model_len=2048,
        enforce_eager=True,
        gpu_memory_utilization=0.7,
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )