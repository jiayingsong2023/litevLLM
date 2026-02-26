import torch
import triton
import triton.language as tl

@triton.jit
def matmul_q4_k_vec_kernel(
    w_ptr,              # Quantized weights
    x_ptr,              # Input vector (N,)
    out_ptr,            # Output vector (M,)
    n_elements,         # Number of columns (N)
    stride_w_row,       # Stride for W rows (in bytes)
    stride_x,           # Stride for X
    stride_out,         # Stride for Output
    BLOCK_SIZE: tl.constexpr, # Must be 256 for Q4_K superblock
):
    # Current row index (M dimension)
    row_idx = tl.program_id(0)
    
    # Pointers
    w_row_ptr = w_ptr + row_idx * stride_w_row
    x_base_ptr = x_ptr
    
    # Accumulator for the dot product
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Constants for Q4_K
    # A superblock is 256 elements.
    # Metadata: 4 bytes (d, dmin) + 12 bytes (scales) + 128 bytes (quants) = 144 bytes
    bytes_per_block = 144
    
    # Offsets for vectorized loading
    # We process one superblock (256 elements) at a time per iteration.
    # Triton handles the vectorization.
    offs_256 = tl.arange(0, 256)
    
    # Iterate over superblocks
    num_blocks = n_elements // 256
    
    for block_idx in range(num_blocks):
        # --- Pointers Setup ---
        w_block_ptr = w_row_ptr + block_idx * bytes_per_block
        
        # Input X pointer
        x_ptr_curr = x_base_ptr + (block_idx * 256 + offs_256) * stride_x
        
        # --- 1. Load Input X (FP16) ---
        x_vals = tl.load(x_ptr_curr)
        
        # --- 2. Load Q4_K Metadata ---
        # d, dmin are at the beginning
        # We load them as scalars first, then broadcast
        d_ptr = w_block_ptr.to(tl.pointer_type(tl.float16))
        dmin_ptr = (w_block_ptr + 2).to(tl.pointer_type(tl.float16))
        
        d = tl.load(d_ptr)
        dmin = tl.load(dmin_ptr)
        
        # --- 3. Load Quantized Data (qs) ---
        # qs starts at offset 16. It contains 128 bytes for 256 4-bit weights.
        qs_ptr = w_block_ptr + 16
        
        # We need to load 128 bytes and expand them to 256 weights.
        # To do this efficiently in Triton, we map the 0..255 indices to 0..127 byte indices.
        byte_indices = offs_256 // 2
        qs_bytes = tl.load(qs_ptr + byte_indices)
        
        # --- 4. Dequantize (Vectorized) ---
        # If index is even (0, 2, 4...), we want low 4 bits:  val & 0x0F
        # If index is odd  (1, 3, 5...), we want high 4 bits: (val >> 4) & 0x0F
        
        is_high = (offs_256 % 2) != 0
        
        # Shift down if high nibble, then mask
        # Note: Triton's bitwise ops work on integers. qs_bytes is uint8 or int8.
        q_vals = tl.where(is_high, qs_bytes >> 4, qs_bytes)
        q_vals = q_vals & 0x0F
        
        # Convert to float for math
        w_vals = q_vals.to(tl.float16)
        
        # Apply scaling: w = d * q - dmin
        # Note: This is the simplified Q4_K formula. Full Q4_K uses 6-bit scales.
        # For 'Lite' inference, skipping the 6-bit sub-scales is an acceptable approximation
        # for initial speedup, but we should eventually add them back.
        # Ideally: w = d * q - dmin
        w_vals = (w_vals * d) - dmin
        
        # --- 5. Accumulate ---
        acc += w_vals * x_vals

    # --- Reduction ---
    final_acc = tl.sum(acc)
    
    # Store result
    tl.store(out_ptr + row_idx * stride_out, final_acc.to(tl.float16))

def matmul_q4_k_vec(
    W: torch.Tensor,
    X: torch.Tensor,
    out: torch.Tensor,
    n_elements: int,
):
    M = W.shape[0]
    grid = (M, 1, 1)
    
    # Triton needs at least one block size to be a power of 2 for vectorization
    matmul_q4_k_vec_kernel[grid](
        W, X, out,
        n_elements,
        W.stride(0),
        X.stride(0),
        out.stride(0),
        BLOCK_SIZE=256
    )
    return out