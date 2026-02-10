# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from gguf import dequantize, GGMLQuantizationType
from vllm.kernels.triton.gguf_dequant import dequant_q4_k_triton

# Global cache to store dequantized weights on GPU
_DEQUANT_CACHE = {}

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
    """
    # Use storage data pointer and offset to uniquely identify the tensor view
    try:
        storage_ptr = W.untyped_storage().data_ptr()
    except AttributeError:
        storage_ptr = W.storage().data_ptr()
        
    cache_key = (storage_ptr, W.storage_offset(), W.shape, quant_type, m, n, dtype)
    
    if cache_key in _DEQUANT_CACHE:
        return _DEQUANT_CACHE[cache_key]

    # Q4_K (Type 12) 使用原生 Triton 内核
    if quant_type == 12:
        res = dequant_q4_k_triton(W, m, n, dtype)
        _DEQUANT_CACHE[cache_key] = res
        return res

    # 其它类型暂时保留 CPU 降级，但同样通过 storage 缓存来加速
    # 只有在第一次遇到该存储切片时才会执行
    w_np = W.cpu().numpy()
    dequant_np = dequantize(w_np, GGMLQuantizationType(quant_type))
    output = torch.from_numpy(dequant_np).to(device=W.device, dtype=dtype)
    res = output.view(m, n)
    _DEQUANT_CACHE[cache_key] = res
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