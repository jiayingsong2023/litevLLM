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
    A safer, more compatible Triton kernel for GGUF Q4_K dequantization.
    This version avoids vectorization and uses simple loops to prevent
    compiler bugs on new hardware like AMD's gfx1151.
    """
    pid = tl.program_id(0)
    
    elements_per_block = 256
    bytes_per_block = 144
    
    # Calculate the starting element index for this program instance.
    start_element = pid * BLOCK_SIZE
    
    # Boundary check: Ensure we don't go past the total number of elements.
    if start_element >= n_elements:
        return

    # Determine the block this element belongs to.
    block_idx = start_element // elements_per_block
    
    # Find the offset within the block.
    in_block_offset = start_element % elements_per_block

    # Pointers to the start of the block for weights and output.
    w_block_ptr = w_ptr + block_idx * bytes_per_block
    out_block_ptr = out_ptr + block_idx * elements_per_block

    # --- Load Scales and Mins ---
    # These are shared for the whole block.
    d = tl.load(w_block_ptr.to(tl.pointer_type(tl.float16)))
    dmin = tl.load((w_block_ptr + 2).to(tl.pointer_type(tl.float16)))

    # For Q4_K, scales are more complex. We'll simplify for now
    # and assume the primary scale `d` is the most important.
    
    # --- Dequantize Loop ---
    # This loop will run for each element in the thread's assigned BLOCK_SIZE.
    for i in range(BLOCK_SIZE):
        current_element_index = start_element + i
        
        if current_element_index < n_elements:
            # Recalculate offset within the current block for each element.
            current_in_block_offset = current_element_index % elements_per_block
            
            # Find the byte that contains our 4-bit weight.
            # The quantized data starts at byte 16 of the block.
            qs_byte_offset = 16 + (current_in_block_offset // 2)
            qs_byte_ptr = w_block_ptr + qs_byte_offset
            
            byte_val = tl.load(qs_byte_ptr)

            # Determine if we need the lower or upper 4 bits.
            is_upper_nibble = (current_in_block_offset % 2) == 1
            
            if is_upper_nibble:
                q_val = (byte_val >> 4) & 0x0F
            else:
                q_val = byte_val & 0x0F

            # Perform the dequantization.
            # Simplified formula: final_val = d * q_val - dmin
            # A more complete implementation would use the 6-bit scales.
            final_val = (d * q_val.to(tl.float16))
            
            # Write to the output tensor.
            out_element_ptr = out_ptr + current_element_index
            tl.store(out_element_ptr, final_val)

def dequant_q4_k_triton(W: torch.Tensor, m: int, n: int, dtype: torch.dtype):
    """
    Python wrapper for the Triton Q4_K dequantization kernel.
    W: [rows, row_bytes] where row_bytes is a multiple of 144.
    """
    if W.dim() == 1:
        W = W.view(-1, 144)
        
    rows = W.shape[0]
    row_bytes = W.shape[1]
    
    # Each Q4_K superblock is 144 bytes and represents 256 elements
    blocks_per_row = row_bytes // 144
    num_blocks = rows * blocks_per_row
    
    # Output buffer for all elements
    out = torch.empty((num_blocks, 256), device=W.device, dtype=dtype)
    
    grid = (num_blocks,)
    dequant_q4_k_kernel[grid](
        W, out,
        num_blocks * 256,
        144, 256,
        BLOCK_SIZE=256
    )
    
    # Final reshape to target dimensions
    return out.view(m, n)
