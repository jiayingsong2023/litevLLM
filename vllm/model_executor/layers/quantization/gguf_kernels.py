# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
import os
from collections import OrderedDict
from gguf import dequantize, GGMLQuantizationType
try:
    from vllm.kernels.triton.gguf_dequant import dequant_q4_k_triton
except ImportError:
    dequant_q4_k_triton = None
try:
    from vllm.kernels.triton.gguf_gemm import matmul_q4_k_vec
except ImportError:
    matmul_q4_k_vec = None
try:
    from vllm.kernels.triton.gguf_gemm import matmul_q4_k_tokens
except ImportError:
    matmul_q4_k_tokens = None

# LRU Cache to store dequantized weights on GPU
# Default size is increased to 300 to reduce cache thrashing on 7B models.
# Llama-2-7B has ~291 tensors. 300 covers the entire model.
# Users can override this via VLLM_GGUF_CACHE_SIZE env var.
_CACHE_CAPACITY = int(os.environ.get("VLLM_GGUF_CACHE_SIZE", "300"))
_DEQUANT_CACHE = OrderedDict()
_ENABLE_Q4K_TRITON_DEQUANT = os.environ.get("FASTINFERENCE_GGUF_Q4K_TRITON_DEQUANT", "0") == "1"
_ENABLE_Q4K_FUSED_GEMM = os.environ.get("FASTINFERENCE_GGUF_Q4K_FUSED", "0") == "1"
_Q4K_FUSED_MAX_TOKENS = int(os.environ.get("FASTINFERENCE_GGUF_Q4K_FUSED_MAX_TOKENS", "4"))

def ggml_dequantize_fallback(
    W: torch.Tensor, 
    quant_type: int, 
    m: int, 
    n: int, 
    dtype: torch.dtype = torch.float16
) -> torch.Tensor:
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

    # Handle MoE 3D tensors [E, rows, cols_bytes]
    orig_shape = W.shape
    if len(W.shape) == 3:
        W_flat = W.view(-1, W.shape[-1])
        m_total = W.shape[0] * m
    else:
        W_flat = W
        m_total = m

    # Q4_K (Type 12) 使用原生 Triton 内核
    if (
        quant_type == 12
        and _ENABLE_Q4K_TRITON_DEQUANT
        and dequant_q4_k_triton is not None
    ):
        res_flat = dequant_q4_k_triton(W_flat, m_total, n, dtype)
    else:
        # print(f"Fallback dequant for type {quant_type} (m_total={m_total}, n={n})")
        w_np = W_flat.cpu().numpy()
        dequant_np = dequantize(w_np, GGMLQuantizationType(quant_type))
        res_flat = torch.from_numpy(dequant_np).to(device=W.device, dtype=dtype)

    res = res_flat.view(orig_shape[0], m, n) if len(orig_shape) == 3 else res_flat.view(m, n)

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

    # Experimental fused path for decode-only/small token batches.
    # Disabled by default to keep strict numerical behavior unchanged.
    if (
        _ENABLE_Q4K_FUSED_GEMM
        and quant_type == 12
        and matmul_q4_k_tokens is not None
        and X.dim() in (1, 2)
        and num_tokens <= _Q4K_FUSED_MAX_TOKENS
        and n % 256 == 0
    ):
        x_tokens = X.view(1, -1) if X.dim() == 1 else X
        out_tokens = torch.empty(
            (x_tokens.shape[0], row),
            dtype=x_tokens.dtype,
            device=x_tokens.device,
        )
        try:
            matmul_q4_k_tokens(W, x_tokens, out_tokens, n)
            return out_tokens if X.dim() == 2 else out_tokens.view(1, row)
        except RuntimeError:
            # Fallback to stable dequant+matmul path.
            pass
    
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
    if X.dim() == 1 or num_tokens == 1:
        return torch.mv(weight, X.view(-1)).view(num_tokens, row)
    return torch.matmul(X, weight.t())

def ggml_mul_mat_a8_fallback(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: int,
) -> torch.Tensor:
    return ggml_mul_mat_vec_a8_fallback(W, X, quant_type, row)