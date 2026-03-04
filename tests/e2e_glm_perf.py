# SPDX-License-Identifier: Apache-2.0
import torch
import time
import os
import json
import gc
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.layers.quantization.gguf import clear_gguf_cache

def benchmark_glm(model_path, batch_size=32):
    print(f"\n>>> Benchmarking GLM (Real Weights): {model_path}")
    print(f">>> Batch Size: {batch_size}")
    
    clear_gguf_cache()
    gc.collect()
    
    # 1. Load config
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        config_data = json.load(f)
    
    class DummyHFConfig:
        def __init__(self, data):
            self.__dict__.update(data)
            # Ensure it matches the registry
            self.architectures = ["Glm4MoeLiteForCausalLM"]
            self.dtype = data.get("torch_dtype", "bfloat16")

    hf_config = DummyHFConfig(config_data)

    class FakeVllmConfig:
        def __init__(self):
            self.model_config = type('obj', (object,), {
                'hf_config': hf_config,
                'dtype': torch.float16,
                'max_model_len': 4096,
                'model': model_path,
                'get_num_kv_heads': lambda x: hf_config.num_key_value_heads,
                'get_head_size': lambda: hf_config.hidden_size // hf_config.num_attention_heads,
                'get_num_layers': lambda x: hf_config.num_hidden_layers,
                'get_total_num_kv_heads': lambda: hf_config.num_key_value_heads,
                'get_max_model_len': lambda: 4096,
            })
            self.parallel_config = type('obj', (object,), {
                'tensor_parallel_size': 1,
                'pipeline_parallel_size': 1,
                'world_size': 1,
            })
            from vllm.model_executor.layers.quantization.gguf import GGUFConfig
            self.quant_config = GGUFConfig()

    v_config = FakeVllmConfig()
    
    # 2. Initialize Model
    print("Loading model...")
    model = get_model(v_config).cuda().half()
    
    # 3. Prepare Inputs
    input_ids = torch.ones((batch_size, 1), device="cuda", dtype=torch.long)
    positions = torch.zeros(batch_size, device="cuda", dtype=torch.long) + 10
    
    num_heads = hf_config.num_attention_heads
    num_kv_heads = hf_config.num_key_value_heads
    head_size = hf_config.hidden_size // num_heads
    
    kv_caches = []
    num_blocks = 128
    for _ in range(hf_config.num_hidden_layers):
        k_cache = torch.zeros((num_blocks, 16, num_kv_heads, head_size), device="cuda", dtype=torch.float16)
        v_cache = torch.zeros((num_blocks, 16, num_kv_heads, head_size), device="cuda", dtype=torch.float16)
        kv_caches.append((k_cache, v_cache))

    attn_metadata = {
        "slot_mapping": torch.arange(batch_size, device="cuda", dtype=torch.int32),
        "seq_lens": torch.ones(batch_size, device="cuda", dtype=torch.int32) * 11,
        "block_tables": torch.zeros((batch_size, num_blocks), device="cuda", dtype=torch.int32)
    }

    # 4. Warmup
    print("Warmup...")
    for _ in range(5):
        with torch.inference_mode():
            model(input_ids, positions, kv_caches, attn_metadata)
    torch.cuda.synchronize()

    # 5. Benchmark
    iters = 20
    print(f"Running {iters} iterations...")
    start_time = time.time()
    for _ in range(iters):
        with torch.inference_mode():
            model(input_ids, positions, kv_caches, attn_metadata)
    torch.cuda.synchronize()
    end_time = time.time()

    avg_latency = (end_time - start_time) / iters * 1000
    total_tps = (1000 / avg_latency) * batch_size
    
    print(f"RESULT: Latency={avg_latency:.2f}ms, Total Throughput={total_tps:.2f} tokens/sec")
    
    del model, kv_caches
    clear_gguf_cache()
    gc.collect()
    
    return total_tps

if __name__ == "__main__":
    model_path = "models/GLM-4.7-Flash-GGUF"
    if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
        results = {}
        for bs in [1, 8, 32]:
            try:
                results[bs] = benchmark_glm(model_path, batch_size=bs)
            except Exception as e:
                print(f"BS={bs} failed: {e}")
                # import traceback
                # traceback.print_exc()

        print("\n" + "="*45)
        print("GLM-4.7-FLASH PERFORMANCE SUMMARY")
        print("="*45)
        for bs, tps in results.items():
            print(f"Batch Size {bs:2}: {tps:8.2f} tokens/sec")
        print("="*45)
    else:
        print(f"Model path not found: {model_path}")
