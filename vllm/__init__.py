# SPDX-License-Identifier: Apache-2.0
import os
# Global memory optimization for AMD/MoE
# Use modern variable name and include garbage collection threshold
if "PYTORCH_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"

from vllm.sampling_params import SamplingParams
from vllm.pooling_params import PoolingParams
from vllm.inputs import TextPrompt, TokensPrompt
from vllm.entrypoints.llm import LLM

__all__ = ["SamplingParams", "PoolingParams", "TextPrompt", "TokensPrompt", "LLM"]
