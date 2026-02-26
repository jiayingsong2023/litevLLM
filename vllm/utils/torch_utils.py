# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import importlib.metadata
import os
import random
import threading
from collections.abc import Callable, Collection
from functools import lru_cache
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import numpy.typing as npt
import torch
from packaging import version
from packaging.version import Version
from torch.library import Library, infer_schema

import vllm.envs as envs
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.sequence import IntermediateTensors
else:
    ModelConfig = object
    IntermediateTensors = object

logger = init_logger(__name__)

STR_DTYPE_TO_TORCH_DTYPE = {
    "float32": torch.float32,
    "half": torch.half,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
    "int8": torch.int8,
    "fp8_inc": torch.float8_e4m3fn,
    "fp8_ds_mla": torch.uint8,
}

TORCH_DTYPE_TO_NUMPY_DTYPE = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.uint8: np.uint8,
    torch.int32: np.int32,
    torch.int64: np.int64,
}

MODELOPT_TO_VLLM_KV_CACHE_DTYPE_MAP = {
    # TODO: Add more modelopt kv cache dtype
    # mappings here when it supported by some attention backend
    # (for example supports nvfp4).
    "fp8": "fp8_e4m3",
}

T = TypeVar("T")

def is_strictly_contiguous(t: torch.Tensor) -> bool:
    if not t.is_contiguous():
        return False

    # Check that strides match canonical contiguous layout
    shape = t.shape
    strides = t.stride()
    expected_stride = 1
    for i in range(len(shape) - 1, -1, -1):
        if strides[i] != expected_stride:
            return False
        expected_stride *= shape[i]
    return True

@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
    Sets the default number of threads for PyTorch to the given value.

    `None` means using the value of the environment variable `OMP_NUM_THREADS`
    (or `1` if that is not available).
    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        yield
        return

    old_value = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        yield
    except Exception as e:
        if "No CUDA GPUs are available" in str(e):
            err_msg = "CUDA initialization is blocked."
        else:
            err_msg = str(e)
        raise RuntimeError(err_msg) from e
    finally:
        if old_value is None:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_value

def get_dtype_size(dtype: torch.dtype) -> int:
    Test whether it is lossless to cast a tensor from
    `src_dtype` to `tgt_dtype`.
    Get the common `dtype` where all of the other `dtypes` can be
    cast to it without losing any information.

    Maps various FP8 format names to vLLM's standard cache dtype strings.
    Returns None if no kv_cache_quant_algo is specified.
    Returns "auto" if the value is not recognized/supported.
    kv_algo_str = get_kv_cache_quant_algo_string(quant_cfg)
    if kv_algo_str is not None and kv_algo_str != "auto":
        # Only convert if we have a valid dtype string (not "auto" fallback)
        return STR_DTYPE_TO_TORCH_DTYPE[kv_algo_str]
    return None

def resolve_kv_cache_dtype_string(
    kv_cache_dtype: str, model_config: ModelConfig
) -> str:
    if kv_cache_dtype != "auto":
        return kv_cache_dtype

    hf_cfg = getattr(model_config, "hf_config", None)
    if hf_cfg is not None:
        quant_cfg = getattr(hf_cfg, "quantization_config", None)
        if quant_cfg is not None:
            kv_algo_str = get_kv_cache_quant_algo_string(quant_cfg)
            if kv_algo_str is not None:
                return kv_algo_str

    # Default to auto (will be handled by downstream code)
    return "auto"

def kv_cache_dtype_str_to_dtype(
    kv_cache_dtype: str, model_config: ModelConfig
) -> torch.dtype:
    if kv_cache_dtype == "auto":
        # Model config may not be specified for unit tests, default to float16
        return model_config.dtype if model_config else torch.half
    return STR_DTYPE_TO_TORCH_DTYPE[kv_cache_dtype]

def set_random_seed(seed: int | None) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

def create_kv_caches_with_random_flash(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: str | torch.dtype | None,
    model_dtype: str | torch.dtype | None = None,
    seed: int | None = None,
    device: str | None = "cuda",
    cache_layout: str | None = "NHD",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    set_random_seed(seed)

    dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)
    generic_kv_cache_shape = (num_blocks, 2, block_size, num_heads, head_size)
    assert cache_layout in ("NHD", "HND")
    stride_order = (0, 1, 2, 3, 4) if cache_layout == "NHD" else (0, 1, 3, 2, 4)

    kv_cache_allocation_shape = tuple(generic_kv_cache_shape[i] for i in stride_order)
    scale = head_size**-0.5

    key_caches: list[torch.Tensor] = []
    value_caches: list[torch.Tensor] = []

    for _ in range(num_layers):
        key_value_cache = torch.empty(
            size=kv_cache_allocation_shape, dtype=dtype, device=device
        ).permute(*stride_order)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            key_value_cache.uniform_(-scale, scale)
        elif cache_dtype == "fp8":
            _generate_random_fp8(key_value_cache, -scale, scale)
        else:
            raise ValueError(f"Does not support key cache of type {cache_dtype}")
        key_caches.append(key_value_cache[:, 0])
        value_caches.append(key_value_cache[:, 1])
    return key_caches, value_caches

def create_kv_caches_with_random(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: str | torch.dtype | None,
    model_dtype: str | torch.dtype | None = None,
    seed: int | None = None,
    device: str | None = "cuda",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if cache_dtype == "fp8" and head_size % 16:
        raise ValueError(
            f"Does not support key cache of type fp8 with head_size {head_size}"
        )

    set_random_seed(seed)

    dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)

    scale = head_size**-0.5
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_caches: list[torch.Tensor] = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape, dtype=dtype, device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            key_cache.uniform_(-scale, scale)
        elif cache_dtype == "fp8":
            _generate_random_fp8(key_cache, -scale, scale)
        else:
            raise ValueError(f"Does not support key cache of type {cache_dtype}")
        key_caches.append(key_cache)

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_caches: list[torch.Tensor] = []
    for _ in range(num_layers):
        value_cache = torch.empty(size=value_cache_shape, dtype=dtype, device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            value_cache.uniform_(-scale, scale)
        elif cache_dtype == "fp8":
            _generate_random_fp8(value_cache, -scale, scale)
        else:
            raise ValueError(f"Does not support value cache of type {cache_dtype}")
        value_caches.append(value_cache)
    return key_caches, value_caches

def async_tensor_h2d(
    data: list,
    dtype: torch.dtype,
    target_device: str | torch.device,
    pin_memory: bool,
) -> torch.Tensor:
    Make a padded array from 2D inputs.

    The padding is applied to the end of each inner list until it reaches
    `max_len`.
    Make a padded tensor from 2D inputs.

    The padding is applied to the end of each inner list until it reaches
    `max_len`.
    replace `torch.cuda.current_stream()` with `vllm.utils.current_stream()`.
    it turns out that `torch.cuda.current_stream()` is quite expensive,
    as it will construct a new stream object at each call.
    here we patch `torch.cuda.set_stream` to keep track of the current stream
    directly, so that we can avoid calling `torch.cuda.current_stream()`.

    the underlying hypothesis is that we do not call `torch._C._cuda_setStream`
    from C/C++ code.
    Ensures aux_stream is initialized only once
    CUDA_VISIBLE_DEVICES at the time of call.

    This should be used instead of torch.cuda.device_count()
    unless CUDA_VISIBLE_DEVICES has already been set to the desired
    Create a weak reference to a tensor.
    The new tensor will share the same data as the original tensor,
    but will not keep the original tensor alive.
    This ignores 0-size tensors as those don't allocate any memory.
    Convenience function to create weak references to tensors,
    for single tensor, list of tensors or tuple of tensors.
    Get a CUDA view of a CPU tensor using Unified Virtual Addressing (UVA).

    Args:
        target: a version string, like "2.6.0".

    Returns:
        Whether the condition meets.

    Args:
        target: a version string, like "2.6.0".

    Returns:
        Whether the condition meets.
    `torch.library.custom_op` can have significant overhead because it
    needs to consider complicated dispatching logic. This function
    directly registers a custom op and dispatches it to the CUDA backend.
    See https://gist.github.com/youkaichao/ecbea9ec9fc79a45d2adce1784d7a9a5
    for more details.

    By default, the custom op is registered to the vLLM library. If you
    want to register it to a different library, you can pass the library
    object to the `target_lib` argument.

    IMPORTANT: the lifetime of the operator is tied to the lifetime of the
    library object. If you want to bind the operator to a different library,
    make sure the library object is alive when the operator is used.
