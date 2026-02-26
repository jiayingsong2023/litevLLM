# SPDX-License-Identifier: Apache-2.0
import torch
import time
import numpy as np
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.llama import LlamaLayer

def benchmark_module(name, module, input_data, iterations=100):
    # Warmup
    for _ in range(10):
        module(*input_data)
    torch.cuda.synchronize()
    
    start_time = time.time()
    for _ in range(iterations):
        module(*input_data)
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / iterations * 1000 # ms
    print(f"{name:20} | Avg Latency: {avg_latency:8.4f} ms")
    return avg_latency

def run_perf_test():
    device = "cuda"
    batch_size = 1
    seq_len = 1
    hidden_size = 4096
    intermediate_size = 11008
    
    print(f"--- LitevLLM Micro-Benchmark (Batch={batch_size}, Hidden={hidden_size}) ---")
    
    # 1. Test LiteLinear
    x = torch.randn(batch_size, hidden_size, device=device)
    linear = LiteLinear(hidden_size, intermediate_size, bias=False).to(device)
    benchmark_module("LiteLinear", linear, (x,))
    
    # 2. Test RMSNorm
    norm = RMSNorm(hidden_size).to(device)
    benchmark_module("RMSNorm", norm, (x,))
    
    # 3. Test LlamaLayer (Full Block)
    class DummyConfig:
        def __init__(self):
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.num_attention_heads = 32
            self.num_key_value_heads = 8
            self.rms_norm_eps = 1e-6
            self.head_dim = 128
    
    config = DummyConfig()
    layer = LlamaLayer(config, layer_id=0).to(device)
    
    positions = torch.zeros(batch_size, dtype=torch.long, device=device)
    # Mock KV cache: [num_layers, num_heads, head_dim, block_size] simplified for test
    kv_cache = torch.randn(1, 8, 128, 128, device=device) 
    attn_metadata = None # Simplified
    
    benchmark_module("LlamaLayer (Full)", layer, (x, positions, kv_cache, attn_metadata))

if __name__ == "__main__":
    if torch.cuda.is_available():
        run_perf_test()
    else:
        print("CUDA not available, skipping perf test.")
