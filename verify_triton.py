import torch
import triton
import triton.language as tl
import sys

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def verify_triton():
    print("\n--- Testing Triton Kernel ---")
    size = 1024
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Simple grid lambda
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    try:
        add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
        if torch.allclose(output, x + y):
            print("Triton kernel verification: SUCCESS")
        else:
            print("Triton kernel verification: FAILED (Numerical mismatch)")
    except Exception as e:
        print(f"Triton kernel verification: FAILED (Error: {e})")

if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"Torch version: {torch.__version__}")
    print(f"ROCm version: {getattr(torch.version, 'hip', 'Not found')}")
    
    if torch.cuda.is_available():
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"GCN Arch: {torch.cuda.get_device_properties(0).gcnArchName}")
    else:
        print("CUDA/ROCm not available in Torch!")
        
    print(f"Triton version: {triton.__version__}")
    
    verify_triton()
