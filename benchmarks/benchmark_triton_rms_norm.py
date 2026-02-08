
import torch
import time
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.config import VllmConfig
from vllm.config.vllm import set_current_vllm_config

def benchmark_rms_norm():
    print("Benchmarking RMSNorm: Triton vs Native...")
    if not torch.cuda.is_available():
        print("Skipping: CUDA not available")
        return

    # Setup dummy config
    config = VllmConfig()
    
    with set_current_vllm_config(config):
        torch.manual_seed(0)
        
        # Dimensions
        batch_sizes = [1, 16, 128]
        hidden_sizes = [1024, 4096, 8192]
        
        epsilon = 1e-6
        dtype = torch.float16
        device = "cuda"

        print(f"{'Batch':<8} {'Hidden':<8} {'Native (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10}")
        print("-" * 60)

        for bs in batch_sizes:
            for hs in hidden_sizes:
                layer = RMSNorm(hs, eps=epsilon).to(device=device, dtype=dtype)
                x = torch.randn(bs, hs, device=device, dtype=dtype)
                
                # Warmup
                for _ in range(10):
                    layer.forward_native(x)
                    layer.forward_cuda(x)
                torch.cuda.synchronize()

                # Native
                start = time.time()
                for _ in range(100):
                    layer.forward_native(x)
                torch.cuda.synchronize()
                native_time = (time.time() - start) / 100 * 1000 # ms

                # Triton
                start = time.time()
                for _ in range(100):
                    layer.forward_cuda(x)
                torch.cuda.synchronize()
                triton_time = (time.time() - start) / 100 * 1000 # ms

                print(f"{bs:<8} {hs:<8} {native_time:<15.4f} {triton_time:<15.4f} {native_time/triton_time:<10.2f}")

if __name__ == "__main__":
    benchmark_rms_norm()
