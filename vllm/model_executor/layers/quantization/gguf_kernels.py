# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
import os
from collections import OrderedDict
from gguf import dequantize, GGMLQuantizationType
from vllm.kernels.triton.gguf_dequant import dequant_q4_k_triton
from vllm.kernels.triton.gguf_gemm import matmul_q4_k_vec

# LRU Cache to store dequantized weights on GPU
# Default size is increased to 128 to reduce cache thrashing on 7B models.
# Llama-2-7B has ~224 weights. 128 covers >50% of the model.
# Users can override this via VLLM_GGUF_CACHE_SIZE env var.
_CACHE_CAPACITY = int(os.environ.get("VLLM_GGUF_CACHE_SIZE", "128"))
_DEQUANT_CACHE = OrderedDict()

def ggml_dequantize_fallback(
    W: torch.Tensor, 
    quant_type: int, 
    m: int, 
    n: int, 
    dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """
    Lite fallback for GGML/GGUF dequantization.
    Caches the result on GPU based on the underlying storage to handle views correctly.
    Uses LRU eviction to manage GPU memory usage.
    """
    global _DEQUANT_CACHE
    
    # Use storage data pointer and offset to uniquely identify the tensor view
    try:
        storage_ptr = W.untyped_storage().data_ptr()
    except AttributeError:
        storage_ptr = W.storage().data_ptr()
        
    cache_key = (storage_ptr, W.storage_offset(), W.shape, quant_type, m, n, dtype)
    
    if cache_key in _DEQUANT_CACHE:
        # Move to end (most recently used)
        _DEQUANT_CACHE.move_to_end(cache_key)
        return _DEQUANT_CACHE[cache_key]

    # Q4_K (Type 12) 使用原生 Triton 内核
    if quant_type == 12:
        res = dequant_q4_k_triton(W, m, n, dtype)
    else:
        # 其它类型暂时保留 CPU 降级
        w_np = W.cpu().numpy()
        dequant_np = dequantize(w_np, GGMLQuantizationType(quant_type))
        output = torch.from_numpy(dequant_np).to(device=W.device, dtype=dtype)
        res = output.view(m, n)

    # Add to cache and evict if necessary
    _DEQUANT_CACHE[cache_key] = res
    if len(_DEQUANT_CACHE) > _CACHE_CAPACITY:
        _DEQUANT_CACHE.popitem(last=False)  # Remove first (least recently used)
        
    return res

def ggml_mul_mat_vec_a8_fallback(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: int,
) -> torch.Tensor:
    """
    Lite fallback for GGML matrix-vector multiplication.
    Dequantizes the weight matrix then performs standard matmul.
    """
    # row is the output dimension (m)
    # W shape is [row, encoded_n]
    # We need to calculate n from W.shape and quant_type
    import gguf
    block_size, type_size = gguf.GGML_QUANT_SIZES[quant_type]
    n = W.shape[1] // type_size * block_size
    
    # Optimization 1: Fused Quant-GEMM for Q4_K
    # Only applies if X is effectively a vector or small batch (which we treat as vec loop)
    # Check if X is [tokens, hidden] and tokens is small
    num_tokens = X.numel() // n
    
    # NOTE: The Fused Kernel implementation is currently functional but slow (~1.6 t/s).
    # We disable it for now to favor the Dequant+Matmul path (~30 t/s) which is
    # now protected by the LRU cache to prevent OOM.
    # if quant_type == 12 and num_tokens == 1:
    #     # Fused Kernel Path
    #     out = torch.empty((row,), dtype=X.dtype, device=X.device)
    #     matmul_q4_k_vec(W, X.flatten(), out, n)
    #     return out.view(num_tokens, row)
    
    weight = ggml_dequantize_fallback(W, quant_type, row, n, X.dtype)
    # X shape [num_tokens, n] or [n]
    # weight shape [row, n]
    return torch.matmul(X, weight.t())

def ggml_mul_mat_a8_fallback(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: int,
) -> torch.Tensor:
    """
    Lite fallback for GGML matrix-matrix multiplication.
    """
    return ggml_mul_mat_vec_a8_fallback(W, X, quant_type, row)