# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from collections import defaultdict
from dataclasses import dataclass, field

import torch

from vllm.config import CacheConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.utils import extract_layer_index
from vllm.multimodal.registry import MultiModalRegistry
from vllm.platforms import current_platform
from vllm.utils.mem_utils import MemorySnapshot, format_gib
from vllm.attention.backend import AttentionBackend, AttentionMetadataBuilder
from vllm.core.encoder_cache_manager import compute_mm_encoder_budget
from vllm.kv_cache_interface import KVCacheGroupSpec, KVCacheSpec

logger = init_logger(__name__)

class MultiModalBudget:
    Perform sanity checks for the result of
    [`vllm.model_executor.models.SupportsMultiModal.embed_multimodal`][].
    Calculate the amount of memory required by vLLM, then validate
    that the current amount of free memory is sufficient for that.
    Sets up KV cache sharing by reusing the allocated KV caches in `kv_caches`
    for layers that do not allocate its own KV cache, based on the mapping in
    `shared_kv_cache_layers`. Adds these layers to the corresponding KV cache
    group, which is needed to ensure that attention metadata is assigned later.

    Args:
        shared_kv_cache_layers: Layer pairings for cross-layer KV sharing.
            If an Attention layer `layer_name` is in the keys of this dict, it
            means this layer will perform attention using the keys and values
            from the KV cache of `shared_kv_cache_layers[layer_name]`.
        kv_cache_groups: The KV cache groups of the model.
    Bind the allocated KV cache to both ModelRunner and forward context so
    that the KV cache can be used in the forward pass.

    This function:
      1) Fills the ModelRunner's kv cache list (`runner_kv_caches`) with
         kv_caches.
      2) Associates each attention layer in the `forward_context` with its
         corresponding KV cache in kv_caches.

    Args:
        kv_caches: The allocated kv_caches with layer names as keys.
        forward_context: The global forward context containing all Attention
            layers with layer names as keys.
        runner_kv_caches: The kv_cache declared by ModelRunner.

    The residual tensor is scattered across tensor parallel ranks when sequence
    parallelism and tensor parallelism is enabled.

    This follows the same logic as SequenceParallelismPass.is_applicable_for_range():
    - In full-graph compilation mode (no splitting ops or using inductor graph
      partition), SP is always applied
    - Otherwise, SP is only applied for specific shapes in compile_sizes
