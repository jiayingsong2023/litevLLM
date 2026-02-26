# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import cache
from typing import NamedTuple, cast, get_args

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.attention.backend import AttentionBackend, AttentionType
from vllm.attention.backends.registry import (
    MAMBA_TYPE_TO_BACKEND_MAP,
    MambaAttentionBackendEnum,
)

logger = init_logger(__name__)

class AttentionSelectorConfig(NamedTuple):
    head_size: int
    dtype: torch.dtype
    kv_cache_dtype: CacheDType | None
    block_size: int | None
    use_mla: bool = False
    has_sink: bool = False
    use_sparse: bool = False
    use_mm_prefix: bool = False
    attn_type: str = AttentionType.DECODER

    def __repr__(self):
        return (
            f"AttentionSelectorConfig(head_size={self.head_size}, "
            f"dtype={self.dtype}, "
            f"kv_cache_dtype={self.kv_cache_dtype}, "
            f"block_size={self.block_size}, "
            f"use_mla={self.use_mla}, "
            f"has_sink={self.has_sink}, "
            f"use_sparse={self.use_sparse}, "
            f"use_mm_prefix={self.use_mm_prefix}, "
            f"attn_type={self.attn_type})"
        )

def get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str | None,
    block_size: int | None,
    use_mla: bool = False,
    has_sink: bool = False,
    use_sparse: bool = False,
    use_mm_prefix: bool = False,
    attn_type: str | None = None,
) -> type[AttentionBackend]:
    return _cached_get_mamba_attn_backend(mamba_type)

@cache
def _cached_get_mamba_attn_backend(
    mamba_type: str,
) -> type[AttentionBackend]:
    assert mamba_type and isinstance(mamba_type, str)

    selected_backend = None
    try:
        backend_name = MAMBA_TYPE_TO_BACKEND_MAP[mamba_type]
        selected_backend = MambaAttentionBackendEnum[backend_name]
    except KeyError as e:
        raise ValueError(
            f"Invalid mamba attention backend type: '{backend_name}'. Valid "
            f"backends are: {list(MambaAttentionBackendEnum.__members__.keys())}"
        ) from e

    mamba_attn_backend = selected_backend.get_class()
    return mamba_attn_backend
