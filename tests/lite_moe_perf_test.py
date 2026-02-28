# SPDX-License-Identifier: Apache-2.0
import torch
import time
from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe

def benchmark_moe_layer():
    print("--- LitevLLM MoE Layer Benchmark ---")
    
    # Qwen1.5-MoE-A2.7B Hyperparameters (Simplified)
    num_tokens = 1
    hidden_size = 2048
    intermediate_size = 1024
    num_experts = 64
    topk = 4
    
    # 1. Prepare Tensors
    hidden_states = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.float16)
    w1 = torch.randn(num_experts, intermediate_size, hidden_size, device="cuda", dtype=torch.float16)
    w2 = torch.randn(num_experts, hidden_size, intermediate_size, device="cuda", dtype=torch.float16)
    gating_output = torch.randn(num_tokens, num_experts, device="cuda", dtype=torch.float16)
    
    # 2. Warmup
    print("Warmup...")
    for _ in range(10):
        fused_moe(hidden_states, w1, w2, gating_output, topk)
    torch.cuda.synchronize()
    
    # 3. Benchmark
    iters = 100
    print(f"Benchmarking {iters} iterations...")
    start_time = time.time()
    for _ in range(iters):
        fused_moe(hidden_states, w1, w2, gating_output, topk)
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / iters * 1000
    print(f"\nAvg MoE Layer Latency: {avg_latency:.2f} ms")
    print(f"Estimated TPS contribution: {1000/avg_latency:.2f} tokens/sec")
    print("-------------------------------------")

if __name__ == "__main__":
    benchmark_moe_layer()
