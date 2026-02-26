# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from copy import copy

import numpy as np
import torch

from vllm.config import CacheConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.utils.math_utils import cdiv
from vllm.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    AttentionType,
    CommonAttentionMetadata,
    subclass_attention_backend_with_overrides,
)
from vllm.attention.selector import get_attn_backend
from vllm.kv_cache_interface import CrossAttentionSpec, KVCacheSpec

logger = init_logger(__name__)

def _get_cross_slot_mapping(
    encoder_seq_lens: np.ndarray,
    block_table_tensor: torch.Tensor,
    kv_cache_spec: CrossAttentionSpec,
    device: torch.device,
) -> torch.Tensor:
    Cross-attention for encoder-decoder models.
    Handles attention between decoder queries and encoder keys/values.
