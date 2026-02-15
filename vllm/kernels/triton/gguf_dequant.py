# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

@triton.jit

def dequant_q4_k_kernel(

    w_ptr,

    out_ptr,

    num_superblocks,

    BLOCK_SIZE: tl.constexpr, # Number of superblocks per program

):

    """

    Vectorized Triton kernel for GGUF Q4_K dequantization.

    Processes BLOCK_SIZE superblocks (each 256 elements) per program instance.

    """

    pid = tl.program_id(0)

    

    # Constants for Q4_K

    elements_per_sb = 256

    bytes_per_sb = 144

    

    # Each program handles BLOCK_SIZE superblocks

    sb_start = pid * BLOCK_SIZE

    sb_end = tl.minimum(sb_start + BLOCK_SIZE, num_superblocks)

    

    for sb_idx in range(sb_start, sb_end):

        # Offsets for this superblock

        w_sb_ptr = w_ptr + sb_idx * bytes_per_sb

        out_sb_ptr = out_ptr + sb_idx * elements_per_sb

        

        # --- 1. Load Metadata (d, dmin) ---

        # d and dmin are the first 4 bytes (2x float16)

        d = tl.load(w_sb_ptr.to(tl.pointer_type(tl.float16)))

        dmin = tl.load((w_sb_ptr + 2).to(tl.pointer_type(tl.float16)))

        

        # --- 2. Load Quantized Data ---

        # The 4-bit quants start at offset 16.

        qs_ptr = w_sb_ptr + 16

        

        # We create a range 0..255 for the output elements

        offs = tl.arange(0, 256)

        

        # Map output index to byte index

        byte_indices = offs // 2

        

        # Load 256 values

        qs_vals = tl.load(qs_ptr + byte_indices)

        

        # --- 3. Unpack 4-bit values ---

        is_high = (offs % 2) == 1

        shifted_vals = tl.where(is_high, qs_vals >> 4, qs_vals)

        q_vals = shifted_vals & 0x0F

        

        # --- 4. Dequantize ---

        res_vals = q_vals.to(tl.float16) * d - dmin

        

        # --- 5. Store Result ---

        tl.store(out_sb_ptr + offs, res_vals)



def dequant_q4_k_triton(W: torch.Tensor, m: int, n: int, dtype: torch.dtype):

    """

    Python wrapper for the Triton Q4_K dequantization kernel.

    """

    total_bytes = W.numel() * W.element_size()

    num_superblocks = total_bytes // 144

    

    out = torch.empty((num_superblocks * 256,), device=W.device, dtype=dtype)

    

    # Tuning: process 16 superblocks per program to balance launch overhead and parallelism

    # Total programs = ceil(num_superblocks / 16)

    sb_per_prog = 16

    grid = (triton.cdiv(num_superblocks, sb_per_prog),)

    

    dequant_q4_k_kernel[grid](

        W, out,

        num_superblocks,

        BLOCK_SIZE=sb_per_prog

    )

    

    return out.view(m, n)
