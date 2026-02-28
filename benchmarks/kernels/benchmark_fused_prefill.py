# SPDX-License-Identifier: Apache-2.0
import torch
import time
from vllm.kernels.triton.reshape_and_cache import reshape_and_cache
from vllm.kernels.triton.fused_attention import fused_prefill_attention

def benchmark_fused_vs_unfused():
    print("--- Benchmark: Fused Prefill Attention vs Unfused ---")
    
    # 模拟一个较大的 Prefill 负载 (例如 1024 tokens)
    num_tokens = 1024
    num_heads = 32
    head_dim = 128
    num_kv_heads = 32
    block_size = 16
    num_blocks = (num_tokens // block_size) + 1
    
    # 1. 准备输入数据
    q = torch.randn((num_tokens, num_heads, head_dim), device="cuda", dtype=torch.float16)
    k = torch.randn((num_tokens, num_kv_heads, head_dim), device="cuda", dtype=torch.float16)
    v = torch.randn((num_tokens, num_kv_heads, head_dim), device="cuda", dtype=torch.float16)
    
    # 2. 准备物理 Cache
    k_cache = torch.zeros((num_blocks, block_size, num_kv_heads, head_dim), device="cuda", dtype=torch.float16)
    v_cache = torch.zeros((num_blocks, block_size, num_kv_heads, head_dim), device="cuda", dtype=torch.float16)
    
    # 3. 准备槽位映射
    slot_mapping = torch.arange(0, num_tokens, device="cuda", dtype=torch.int32)
    
    # 输出缓存
    out_unfused = torch.empty_like(q)
    out_fused = torch.empty_like(q)
    
    # --- 测试 Unfused 模式 (2 个独立 Kernel) ---
    def run_unfused():
        # 1. 写 Cache
        reshape_and_cache(k, v, k_cache, v_cache, slot_mapping, "float16")
        # 2. 模拟计算 (使用 Torch 乘法)
        torch.matmul(q.unsqueeze(2), k.unsqueeze(3)) 
    
    # --- 测试 Fused 模式 (1 个融合 Kernel) ---
    def run_fused():
        fused_prefill_attention(q, k, v, k_cache, v_cache, slot_mapping, out_fused, 0.125)

    # 预热
    print("Warming up...")
    for _ in range(10):
        run_unfused()
        run_fused()
    torch.cuda.synchronize()

    # 测量 Unfused
    iters = 50
    start = time.time()
    for _ in range(iters):
        run_unfused()
    torch.cuda.synchronize()
    unfused_time = (time.time() - start) / iters * 1000
    
    # 测量 Fused
    start = time.time()
    for _ in range(iters):
        run_fused()
    torch.cuda.synchronize()
    fused_time = (time.time() - start) / iters * 1000

    print(f"\n[Results] Tokens: {num_tokens}, Heads: {num_heads}")
    print(f"Unfused Latency: {unfused_time:.4f} ms")
    print(f"Fused Latency:   {fused_time:.4f} ms")
    print(f"Speedup:         {unfused_time / fused_time:.2f}x")
    print("--------------------------------------------------")

if __name__ == "__main__":
    if torch.cuda.is_available():
        benchmark_fused_vs_unfused()
    else:
        print("CUDA not available.")
