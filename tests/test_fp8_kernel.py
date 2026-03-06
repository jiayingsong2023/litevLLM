# SPDX-License-Identifier: Apache-2.0
import torch
import time
from vllm.kernels.triton.fp8_gemm import fp8_block_gemm

def test_fp8_block_gemm():
    M, N, K = 1024, 1024, 1024
    device = "cuda"
    
    BLOCK_M, BLOCK_K = 64, 64
    
    try:
        a_fp16 = torch.randn((M, K), device=device, dtype=torch.float16)
        b_fp16 = torch.randn((K, N), device=device, dtype=torch.float16)
        
        # Block-wise scales [M//64, K//64]
        scale_a = torch.ones((M // 64, K // 64), device=device, dtype=torch.float32) * 0.5
        scale_b = torch.ones((N // 64, K // 64), device=device, dtype=torch.float32) * 2.0
        
        a_fp8 = a_fp16.to(torch.float8_e4m3fn)
        b_fp8 = b_fp16.to(torch.float8_e4m3fn)
        
        print(f"FP8 Block GEMM Test: M={M}, N={N}, K={K}")
        
        # 2. Run Triton Kernel
        out_triton = fp8_block_gemm(a_fp8, b_fp8, scale_a, scale_b)
        
        # 3. Run Reference
        # For simplicity, since scales are uniform 1.0 in this test, it's just matmul
        out_ref = torch.matmul(a_fp16, b_fp16) * (0.5 * 2.0)
        
        # 4. Compare
        cos_sim = torch.nn.functional.cosine_similarity(out_triton.flatten(), out_ref.flatten(), dim=0)
        print(f"Cosine Similarity: {cos_sim.item():.4f}")
        
        # 5. Benchmark
        iters = 100
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            fp8_block_gemm(a_fp8, b_fp8, scale_a, scale_b, out=out_triton)
        torch.cuda.synchronize()
        t1 = time.time()
        
        latency = (t1 - t0) / iters * 1000
        tflops = 2 * M * N * K / (latency / 1000) / 1e12
        print(f"Latency: {latency:.3f} ms, Performance: {tflops:.2f} TFLOPS")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fp8_block_gemm()
