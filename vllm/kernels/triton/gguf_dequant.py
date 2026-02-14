# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

@triton.jit
def dequant_q4_k_kernel(
    w_ptr,
    out_ptr,
    n_elements,
    stride_w_row,
    stride_out_row,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Vectorized Triton kernel for GGUF Q4_K dequantization.
    Processes one superblock (256 elements) per program instance.
    """
    pid = tl.program_id(0)
    
    # Constants for Q4_K
    elements_per_block = 256
    bytes_per_block = 144
    
    # Offsets for this block
    # We are processing block 'pid'.
    # w_ptr is the base of the entire weight matrix (flattened logic handled in wrapper)
    w_block_ptr = w_ptr + pid * bytes_per_block
    out_block_ptr = out_ptr + pid * elements_per_block
    
    # --- 1. Load Metadata (d, dmin) ---
    # d and dmin are the first 4 bytes (2x float16)
    # We load them as scalars.
    d = tl.load(w_block_ptr.to(tl.pointer_type(tl.float16)))
    dmin = tl.load((w_block_ptr + 2).to(tl.pointer_type(tl.float16)))
    
    # --- 2. Load Quantized Data ---
    # The 4-bit quants start at offset 16. There are 128 bytes representing 256 values.
    qs_ptr = w_block_ptr + 16
    
    # We create a range 0..255 for the output elements
    offs = tl.arange(0, 256)
    
    # Map output index to byte index: byte_idx = idx // 2
    # This generates indices [0, 0, 1, 1, 2, 2, ..., 127, 127]
    byte_indices = offs // 2
    
    # Load 256 values (gathering from 128 bytes). 
    # Note: On simple implementations this is a gather load. 
    # More optimized versions might load 128 bytes then expand, but this is concise.
    qs_vals = tl.load(qs_ptr + byte_indices)
    
    # --- 3. Unpack 4-bit values ---
    # Low nibble if even index, High nibble if odd index
    is_high = (offs % 2) == 1
    
    # Shift right if high nibble
    shifted_vals = tl.where(is_high, qs_vals >> 4, qs_vals)
    
    # Mask to get lower 4 bits
    q_vals = shifted_vals & 0x0F
    
    # --- 4. Dequantize ---
    # w = d * q - dmin
    # Convert to float16 for calculation
    w_vals = q_vals.to(tl.float16)
    res_vals = w_vals * d - dmin
    
    # --- 5. Store Result ---
    # Write 256 float16 values contiguously
    tl.store(out_block_ptr + offs, res_vals)

def dequant_q4_k_triton(W: torch.Tensor, m: int, n: int, dtype: torch.dtype):
    """
    Python wrapper for the Triton Q4_K dequantization kernel.
    W: [rows, row_bytes] where row_bytes is a multiple of 144.
    """
    # Flatten W to handle the grid simply
    # Total Q4_K superblocks = total_elements / 256
    # Each superblock is 144 bytes in W
    
    # Calculate total superblocks based on W's raw byte size
    total_bytes = W.numel() * W.element_size() # W is usually byte tensor or uint8
    num_blocks = total_bytes // 144
    
    # Output is [m, n], total elements = m * n = num_blocks * 256
    out = torch.empty((num_blocks * 256,), device=W.device, dtype=dtype)
    
    # Launch one program per superblock
    grid = (num_blocks,)
    
    dequant_q4_k_kernel[grid](
        W, out,
        num_blocks * 256, # n_elements unused in this grid logic but passed for compat
        144, 256,
        BLOCK_SIZE=256
    )
    
    return out.view(m, n)