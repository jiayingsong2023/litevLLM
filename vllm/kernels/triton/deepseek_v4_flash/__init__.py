# SPDX-License-Identifier: Apache-2.0
"""Triton kernel package for DeepSeek V4 Flash experimental support."""

from .attention import DeepSeekV4AttentionKernelInputs, deepseek_v4_attention
from .cache import DeepSeekV4CacheUpdateInputs, deepseek_v4_cache_update
from .compressed_attention import (
    DeepSeekV4CompressedAttentionInputs,
    DeepSeekV4CompressedAttentionTensorInputs,
    deepseek_v4_compressed_attention,
)
from .moe import DeepSeekV4MoEKernelInputs, deepseek_v4_moe
from .output import DeepSeekV4OutputKernelInputs, deepseek_v4_output_projection
from .q8_linear import q8_0_linear

__all__ = [
    "DeepSeekV4AttentionKernelInputs",
    "DeepSeekV4CacheUpdateInputs",
    "DeepSeekV4CompressedAttentionInputs",
    "DeepSeekV4CompressedAttentionTensorInputs",
    "DeepSeekV4MoEKernelInputs",
    "DeepSeekV4OutputKernelInputs",
    "deepseek_v4_attention",
    "deepseek_v4_cache_update",
    "deepseek_v4_compressed_attention",
    "deepseek_v4_moe",
    "deepseek_v4_output_projection",
    "q8_0_linear",
]
