# SPDX-License-Identifier: Apache-2.0
import torch
import time
from vllm.kernels.triton.awq_fused_gemm import awq_fused_gemm
from vllm.model_executor.layers.quantization.awq_triton import awq_dequantize_triton

def benchmark_fused_vs_dequant():
    device = "cuda"
    dtype = torch.float16
    M, K, N = 32, 4096, 4096
    group_size = 128
    
    # 1. Generate Random Weights in [N, K // 8] Layout
    qweight = torch.randint(0, 1000000, (N, K // 8), device=device, dtype=torch.int32)
    scales = torch.randn((N, K // group_size), device=device, dtype=dtype)
    qzeros = torch.randint(0, 1000000, (N, K // group_size // 8), device=device, dtype=torch.int32)
    x = torch.randn((M, K), device=device, dtype=dtype)

    print(f"=== AWQ Fused Kernel Benchmark (M={M}, K={K}, N={N}) ===")

    # 2. Dequant + Matmul Path
    def dequant_matmul_path(x, qweight, scales, qzeros):
        # Use existing reference dequantizer
        # Reference expects [N, K//8] for Qwen/standard if GROUP_ALONG_ROW logic is right
        w = awq_dequantize_triton(qweight, scales, qzeros, group_size)
        # w is [N, K]
        return torch.matmul(x, w.t())

    # Warmup
    for _ in range(10):
        _ = dequant_matmul_path(x, qweight, scales, qzeros)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(100):
        res1 = dequant_matmul_path(x, qweight, scales, qzeros)
    torch.cuda.synchronize()
    t_dequant = (time.time() - start) * 10
    print(f"Dequant + Matmul Latency: {t_dequant:.3f} ms")

    # 3. Fused Path
    # Warmup
    for _ in range(10):
        _ = awq_fused_gemm(x, qweight, scales, qzeros, group_size)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(100):
        res2 = awq_fused_gemm(x, qweight, scales, qzeros, group_size)
    torch.cuda.synchronize()
    t_fused = (time.time() - start) * 10
    print(f"Fused Kernel Latency:    {t_fused:.3f} ms")

    # 4. Correctness Check
    diff = (res1 - res2).abs().mean()
    print(f"Mean Difference: {diff:.6f}")
    
    if diff < 1e-2:
        print("✅ Correctness Verified!")
    else:
        print("❌ Significant Difference Detected!")

if __name__ == "__main__":
    benchmark_fused_vs_dequant()
