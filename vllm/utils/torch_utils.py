# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable

import torch


def dtype_nbytes(dtype: torch.dtype) -> int:
    """Return the number of bytes per element for a torch dtype."""
    f8 = getattr(torch, "float8_e4m3fn", None)
    if f8 is not None and dtype == f8:
        return 1
    if dtype == torch.uint8:
        return 1
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    if dtype == torch.float32:
        return 4
    return 2


def direct_register_custom_op(
    op_name: str,
    op_func: Callable,
    mutates_args: list[str],
    fake_impl: Callable | None = None,
):
    # Simplified registration for LitevLLM
    # In Lite architecture, we can often call functions directly,
    # but we provide this shim for compatibility with common Op structures.
    pass


def is_hip():
    return torch.version.hip is not None


def kv_cache_dtype_str_to_dtype(kv_cache_dtype: str) -> torch.dtype:
    if kv_cache_dtype == "auto":
        return torch.float16
    if "fp8" in kv_cache_dtype.lower():
        return torch.float8_e4m3fn
    if "int4" in kv_cache_dtype.lower():
        # We store packed INT4 in uint8
        return torch.uint8
    if kv_cache_dtype == "fp16":
        return torch.float16
    if kv_cache_dtype == "bf16":
        return torch.bfloat16
    return torch.float16
