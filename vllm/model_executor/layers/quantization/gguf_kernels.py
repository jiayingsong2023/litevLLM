# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from gguf import dequantize, GGMLQuantizationType

def ggml_dequantize_fallback(
    W: torch.Tensor, 
    quant_type: int, 
    m: int, 
    n: int, 
    dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """
    Lite fallback for GGML/GGUF dequantization using the gguf-py library.
    """
    # Convert torch tensor to numpy for gguf.dequantize
    w_np = W.cpu().numpy()
    
    # dequantize returns a numpy array
    dequant_np = dequantize(w_np, GGMLQuantizationType(quant_type))
    
    # Convert back to torch and reshape
    # Note: gguf.dequantize output might need reshaping to (m, n)
    # The output of dequantize is typically a 1D or 2D array depending on input
    output = torch.from_numpy(dequant_np).to(device=W.device, dtype=dtype)
    
    return output.view(m, n)

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