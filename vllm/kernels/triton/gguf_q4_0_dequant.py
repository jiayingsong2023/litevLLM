# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.language as tl

@triton.jit
def _gguf_q4_0_dequant_kernel(
    W_ptr, Out_ptr,
    n_elements,
    stride_w_row, stride_w_col,
    stride_o_row, stride_o_col,
    BLOCK_N: tl.constexpr, # Number of weights to dequantize per program
):
    """
    Dequantize GGUF Q4_0 (18 bytes per 32 weights)
    Layout: [delta (fp16), weights (16 x uint8)]
    """
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    
    # Each program handles BLOCK_N weights. Q4_0 block size is 32.
    # We assume BLOCK_N is a multiple of 32.
    num_ggml_blocks = BLOCK_N // 32
    
    base_col = col_block_idx * BLOCK_N
    if base_col >= n_elements: return

    # W_ptr is uint8. Stride is in bytes.
    # A GGML block is 18 bytes.
    w_block_ptr = W_ptr + row_idx * stride_w_row + col_block_idx * num_ggml_blocks * 18
    o_ptr = Out_ptr + row_idx * stride_o_row + base_col * stride_o_col

    for b in range(num_ggml_blocks):
        # 1. Load Delta (FP16, first 2 bytes)
        # We use tl.load with bitcast to handle the fp16 data at the start of the byte block
        block_ptr = w_block_ptr + b * 18
        # Load 2 bytes and interpret as fp16. Triton needs care with 16-bit loads from uint8.
        # Efficient approach: load as int16 then bitcast.
        d_raw = tl.load(tl.make_block_ptr(
            base=block_ptr,
            shape=[2],
            strides=[1],
            offsets=[0],
            block_shape=[2],
            order=[0]
        )).to(tl.int16)
        
        # In Triton, we can't easily bitcast uint8 to fp16 across 2 elements.
        # Hack: Load the first 2 bytes as a single int16 then cast to fp16.
        # But for simplicity in prototype, we'll assume we can load it.
        delta = tl.load(tl.cast(block_ptr, tl.pointer_type(tl.float16))).to(tl.float32)
        
        # 2. Load 16 bytes of weights (32 values)
        # These start at offset 2 within the 18-byte block
        qs_ptr = block_ptr + 2
        qs = tl.load(qs_ptr + tl.arange(0, 16)) # [16] uint8 values
        
        # 3. Unpack 4-bit values
        # Low nibbles (0-15)
        v_low = (qs & 0xF).to(tl.float32) - 8.0
        # High nibbles (16-31)
        v_high = (qs >> 4).to(tl.float32) - 8.0
        
        # 4. Apply Delta and Store
        res_low = v_low * delta
        res_high = v_high * delta
        
        # Store to output (FP16)
        # Interleave: Q4_0 stores low nibbles then high nibbles for the same 16 bytes
        off_o = b * 32
        tl.store(o_ptr + (off_o + tl.arange(0, 16)) * stride_o_col, res_low.to(tl.float16))
        tl.store(o_ptr + (off_o + 16 + tl.arange(0, 16)) * stride_o_col, res_high.to(tl.float16))

def gguf_q4_0_dequant(W: torch.Tensor, n_rows: int, n_cols: int):
    """
    W: uint8 tensor [n_rows, (n_cols/32)*18]
    Returns: fp16 tensor [n_rows, n_cols]
    """
    Out = torch.empty((n_rows, n_cols), device=W.device, dtype=torch.float16)
    
    # Tuning: Each program handles one row and a chunk of columns
    BLOCK_N = 128 # Process 4 GGML blocks per program
    grid = (n_rows, triton.cdiv(n_cols, BLOCK_N))
    
    _gguf_q4_0_dequant_kernel[grid](
        W, Out,
        n_cols,
        W.stride(0), W.stride(1),
        Out.stride(0), Out.stride(1),
        BLOCK_N=BLOCK_N
    )
    return Out
