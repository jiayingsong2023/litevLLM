# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.language as tl

@triton.jit
def awq_dequantize_kernel(
    qweight_ptr, scales_ptr, zeros_ptr, out_ptr,
    n_rows, n_cols, group_size,
    stride_qw, stride_qn,
    stride_s, stride_sn,
    stride_z, stride_zn,
    stride_out, stride_on,
    GROUP_ALONG_ROW: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Generalized AWQ Dequantization Kernel.
    Supports flexible layouts and grouping axes.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    
    # Load packed weights [n_rows, n_cols // 8]
    # n_cols is the dimension being unpacked
    ptr_q = qweight_ptr + offs_k[:, None] * stride_qw + (offs_n[None, :] // 8) * stride_qn
    q_packed = tl.load(ptr_q, (offs_k[:, None] < n_rows) & (offs_n[None, :] < n_cols))
    
    # Unpack 4-bit
    shift = (offs_n[None, :] % 8) * 4
    q = (q_packed >> shift) & 0xF
    
    # Load scales and zeros
    if GROUP_ALONG_ROW:
        # Standard AWQ: groups along the row axis (usually K)
        group_idx = offs_k // group_size
        mask_s = (group_idx[:, None] < (n_rows // group_size)) & (offs_n[None, :] < n_cols)
        ptr_s = scales_ptr + group_idx[:, None] * stride_s + offs_n[None, :] * stride_sn
        ptr_z = zeros_ptr + group_idx[:, None] * stride_z + (offs_n[None, :] // 8) * stride_zn
    else:
        # Qwen Style: groups along the column axis (usually K)
        group_idx = offs_n // group_size
        mask_s = (offs_k[:, None] < n_rows) & (group_idx[None, :] < (n_cols // group_size))
        ptr_s = scales_ptr + offs_k[:, None] * stride_s + group_idx[None, :] * stride_sn
        ptr_z = zeros_ptr + offs_k[:, None] * stride_z + (group_idx[None, :] // 8) * stride_zn
    
    scales = tl.load(ptr_s, mask=mask_s)
    z_packed = tl.load(ptr_z, mask=mask_s)
    z_shift = (group_idx % 8) * 4 if not GROUP_ALONG_ROW else (offs_n[None, :] % 8) * 4
    # Simplification: if GROUP_ALONG_ROW, zeros are packed along N. 
    # If not, zeros are packed along the group dimension.
    # This part is tricky, let's assume symmetric for now if it gets too complex, 
    # but we try to match the common case.
    zeros = (z_packed >> z_shift) & 0xF
    
    # Compute
    res = (q.to(tl.float32) - zeros.to(tl.float32)) * scales.to(tl.float32)
    
    # Store
    ptr_out = out_ptr + offs_k[:, None] * stride_out + offs_n[None, :] * stride_on
    # We cast to float32 first then to the target output dtype to ensure accuracy
    tl.store(ptr_out, res.to(out_ptr.dtype.element_ty), (offs_k[:, None] < n_rows) & (offs_n[None, :] < n_cols))

def awq_dequantize_triton(qweight, scales, zeros, group_size=128, out_dtype=torch.float16):
    # Determine Layout
    # qweight: [R, C/8]
    # scales: [SR, SC]
    
    n_rows = qweight.shape[0]
    n_cols = qweight.shape[1] * 8
    
    # If scales has same rows as qweight, it's grouping along columns (Qwen style)
    if scales.shape[0] == n_rows:
        group_along_row = False
    else:
        group_along_row = True

    output = torch.empty((n_rows, n_cols), device=qweight.device, dtype=out_dtype)
    grid = lambda META: (triton.cdiv(n_cols, META['BLOCK_N']), triton.cdiv(n_rows, META['BLOCK_K']))
    
    awq_dequantize_kernel[grid](
        qweight, scales, zeros, output,
        n_rows, n_cols, group_size,
        qweight.stride(0), qweight.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0), zeros.stride(1),
        output.stride(0), output.stride(1),
        GROUP_ALONG_ROW=group_along_row,
        BLOCK_N=64, BLOCK_K=32
    )
    return output

def awq_gemm_triton(input, qweight, scales, zeros, group_size=128, split_k_iters=1, out_dtype=torch.float16):
    w = awq_dequantize_triton(qweight, scales, zeros, group_size, out_dtype=out_dtype)
    if w.dtype == torch.float8_e4m3fn:
        w = w.to(input.dtype)
    return torch.matmul(input, w.t())
