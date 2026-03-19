import torch
import sys

def test_alloc():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("CUDA NOT AVAILABLE")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    # Try multiple allocation sizes
    sizes = [
        (1024,), # 4KB
        (1024, 1024), # 4MB
        (4096, 16, 12, 64), # 100MB (Qwen-like)
        (4096, 16, 16, 128), # 250MB (DeepSeek-like)
    ]
    
    for s in sizes:
        print(f"Allocating {s}...")
        try:
            # Use empty to avoid filling kernel
            x = torch.empty(s, device="cuda", dtype=torch.float16)
            torch.cuda.synchronize()
            print(f"  Success. Shape: {x.shape}")
            
            print(f"Filling {s} with zeros...")
            x.zero_()
            torch.cuda.synchronize()
            print(f"  Zero Fill Success.")
            
            del x
        except Exception as e:
            print(f"  FAILED: {e}")

if __name__ == "__main__":
    test_alloc()
