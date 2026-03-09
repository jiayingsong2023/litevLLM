# SPDX-License-Identifier: Apache-2.0
import torch
import numpy as np
import os
from collections import OrderedDict
from gguf import dequantize, GGMLQuantizationType

_CACHE_CAPACITY = int(os.environ.get("VLLM_GGUF_CACHE_SIZE", "300"))
_DEQUANT_CACHE = OrderedDict()
_USE_FP8_WEIGHTS = os.environ.get("FASTINFERENCE_GGUF_FP8", "1") == "1"

def ggml_dequantize_fallback(W: torch.Tensor, quant_type: int, m: int, n: int, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    global _DEQUANT_CACHE
    target_dtype = torch.float8_e4m3fn if _USE_FP8_WEIGHTS else dtype
    
    try: storage_ptr = W.untyped_storage().data_ptr()
    except AttributeError: storage_ptr = W.storage().data_ptr()
    cache_key = (storage_ptr, W.storage_offset(), W.shape, quant_type, m, n, target_dtype)
    
    if cache_key in _DEQUANT_CACHE:
        _DEQUANT_CACHE.move_to_end(cache_key); return _DEQUANT_CACHE[cache_key]

    orig_shape = W.shape
    W_flat = W.view(-1, W.shape[-1]) if len(W.shape) == 3 else W
    
    # --- MEMORY OPTIMIZED DEQUANT PATH ---
    # 1. Dequantize on CPU to float32/float16 numpy
    w_np = W_flat.cpu().numpy()
    dequant_np = dequantize(w_np, GGMLQuantizationType(quant_type))
    
    # 2. Convert to Target Dtype ON CPU before moving to GPU
    # This avoids the 26GB temporary FP16 buffer on GPU
    temp_torch = torch.from_numpy(dequant_np)
    if _USE_FP8_WEIGHTS:
        # Manually convert to FP8 on CPU if supported, or use half to minimize GPU peak
        # Since CPU float8 support is limited, we use half as a safer transit
        res_flat = temp_torch.to(dtype=torch.float16).to(target_dtype).to(device=W.device)
    else:
        res_flat = temp_torch.to(device=W.device, dtype=dtype)

    res = res_flat.view(orig_shape[0], m, n) if len(orig_shape) == 3 else res_flat.view(m, n)

    _DEQUANT_CACHE[cache_key] = res
    if len(_DEQUANT_CACHE) > _CACHE_CAPACITY: _DEQUANT_CACHE.popitem(last=False)
    return res

def ggml_mul_mat_vec_a8_fallback(W: torch.Tensor, X: torch.Tensor, quant_type: int, row: int) -> torch.Tensor:
    import gguf
    block_size, type_size = gguf.GGML_QUANT_SIZES[quant_type]
    n = W.shape[1] // type_size * block_size
    num_tokens = X.numel() // n
    
    weight = ggml_dequantize_fallback(W, quant_type, row, n, X.dtype)
    # PyTorch linear expects floating point, cast FP8 back to X.dtype for computation
    working_weight = weight.to(X.dtype) if weight.dtype == torch.float8_e4m3fn else weight
    
    if X.dim() == 1 or num_tokens == 1:
        return torch.mv(working_weight, X.view(-1)).view(num_tokens, row)
    return torch.matmul(X.view(num_tokens, n), working_weight.t())

def ggml_mul_mat_a8_fallback(W: torch.Tensor, X: torch.Tensor, quant_type: int, row: int) -> torch.Tensor:
    return ggml_mul_mat_vec_a8_fallback(W, X, quant_type, row)
