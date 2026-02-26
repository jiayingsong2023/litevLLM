# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
