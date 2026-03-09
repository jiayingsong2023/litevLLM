# SPDX-License-Identifier: Apache-2.0
import os
# Global memory optimization for AMD/MoE
if "PYTORCH_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"

# Enable experimental ROCm Flash/Mem Efficient Attention to remove warnings and boost perf
if "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL" not in os.environ:
    os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

from vllm.sampling_params import SamplingParams
from vllm.pooling_params import PoolingParams
from vllm.inputs import TextPrompt, TokensPrompt
from vllm.entrypoints.llm import LLM

__all__ = ["SamplingParams", "PoolingParams", "TextPrompt", "TokensPrompt", "LLM"]
