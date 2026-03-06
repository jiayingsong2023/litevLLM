# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.language as tl

@triton.jit
def gguf_dequant_kernel(
    q_ptr, scales_ptr, out_ptr,
    stride_q, stride_s, stride_o,
    N, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # Simplified q4_0 dequant logic for Triton
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off < N
    
    # Load 4-bit packed weights (2 per byte)
    q = tl.load(q_ptr + off // 2, mask=mask)
    s = tl.load(scales_ptr + pid, mask=(pid < (N // BLOCK_SIZE)))
    
    # Dequantize
    shift = (off % 2) * 4
    val = (q >> shift) & 0xF
    res = (val.to(tl.float32) - 8.0) * s
    
    tl.store(out_ptr + off, res.to(tl.float16), mask=mask)

def gguf_dequantize(qweight, qscales, qtype="q4_0"):
    """
    Stabilized Dequantization for AMD APU.
    Uses PyTorch native ops for large tensors to avoid hardware exceptions.
    """
    out_features = qweight.shape[0]
    in_features = qweight.shape[1] * 2
    
    # 1. Stability Path: Use PyTorch for large matrices
    if out_features * in_features > 1024 * 1024:
        q_flat = qweight.view(-1)
        
        # Dequantize logic: (byte & 0xF - 8), (byte >> 4 - 8)
        low = (q_flat & 0x0F).to(torch.float16)
        high = (q_flat >> 4).to(torch.float16)
        
        # Interleave bits to match weight order
        res = torch.stack([low, high], dim=1).view(out_features, in_features)
        
        # Apply scales
        return (res - 8.0) * qscales.unsqueeze(-1)
    
    # 2. Performance Path: Triton for smaller matrices
    output = torch.empty((out_features, in_features), device=qweight.device, dtype=torch.float16)
    # Correctly pass META as constexpr
    grid = lambda META: (triton.cdiv(out_features * in_features, META['BLOCK_SIZE']),)
    
    gguf_dequant_kernel[grid](
        qweight, qscales, output,
        in_features // 2, 1, in_features,
        out_features * in_features,
        BLOCK_SIZE=32
    )
    return output


@triton.jit
def dequant_q4_k_kernel(
    w_ptr, out_ptr,
    stride_w_row, stride_out_row,
    n_elements,
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    bytes_per_block = 144
    offs_128 = tl.arange(0, 128)
    out_block_base = block_idx * 256
    mask = out_block_base + offs_128 * 2 < n_elements

    w_row_ptr = w_ptr + row_idx * stride_w_row
    w_block_ptr = w_row_ptr + block_idx * bytes_per_block

    d = tl.load(w_block_ptr.to(tl.pointer_type(tl.float16)))
    dmin = tl.load((w_block_ptr + 2).to(tl.pointer_type(tl.float16)))
    qs = tl.load(w_block_ptr + 16 + offs_128, mask=mask, other=0)

    q_low = (qs & 0x0F).to(tl.float16)
    q_high = ((qs >> 4) & 0x0F).to(tl.float16)

    w_low = (q_low * d) - dmin
    w_high = (q_high * d) - dmin

    out_row_ptr = out_ptr + row_idx * stride_out_row
    out_low_offs = out_block_base + offs_128 * 2
    out_high_offs = out_low_offs + 1
    tl.store(out_row_ptr + out_low_offs, w_low, mask=mask)
    tl.store(out_row_ptr + out_high_offs, w_high, mask=mask)


def dequant_q4_k_triton(
    qweight: torch.Tensor,
    m: int,
    n: int,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Approximate Q4_K dequantization on GPU.
    The current formula mirrors the fast path used by gguf_gemm kernels.
    """
    if qweight.dim() != 2:
        raise RuntimeError("dequant_q4_k_triton expects a 2D packed tensor.")
    if n % 256 != 0:
        raise RuntimeError(f"Q4_K n must be divisible by 256, got {n}.")

    output = torch.empty((m, n), device=qweight.device, dtype=dtype)
    grid = (m, n // 256)
    dequant_q4_k_kernel[grid](
        qweight,
        output,
        qweight.stride(0),
        output.stride(0),
        n,
    )
    return output
