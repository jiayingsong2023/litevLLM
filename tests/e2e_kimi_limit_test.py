# SPDX-License-Identifier: Apache-2.0
import torch
import time
import os
import json
import gc
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.layers.quantization.gguf import clear_gguf_cache

def stress_test_kimi(model_path, batch_size=8, context_len=1024):
    print(f"
>>> [LIMIT TEST] Kimi-Linear-48B Stress Test")
    print(f">>> BS={batch_size}, Ctx={context_len}")
    
    clear_gguf_cache()
    gc.collect()
    torch.cuda.empty_cache()
    
    # 1. Load config
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        config_data = json.load(f)
    
    class DummyHFConfig:
        def __init__(self, data):
            self.__dict__.update(data)
            self.architectures = ["KimiLinearForCausalLM"]
            self.dtype = "bfloat16"

    hf_config = DummyHFConfig(config_data)

    class FakeVllmConfig:
        def __init__(self):
            self.model_config = type('obj', (object,), {
                'hf_config': hf_config,
                'dtype': torch.float16,
                'max_model_len': context_len + 128,
                'model': model_path,
                'get_num_kv_heads': lambda x: hf_config.num_key_value_heads,
                'get_head_size': lambda: hf_config.hidden_size // hf_config.num_attention_heads,
                'get_num_layers': lambda x: hf_config.num_hidden_layers,
                'get_total_num_kv_heads': lambda: hf_config.num_key_value_heads,
                'get_max_model_len': lambda: context_len + 128,
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
    print("Loading 48B model weights (GGUF Optimized)...")
    model = get_model(v_config).cuda().half()
    
    # 3. Prepare Inputs
    input_ids = torch.ones((batch_size, 1), device="cuda", dtype=torch.long)
    positions = torch.zeros(batch_size, device="cuda", dtype=torch.long) + (context_len - 1)
    
    num_heads = hf_config.num_attention_heads
    num_kv_heads = hf_config.num_key_value_heads
    head_size = hf_config.hidden_size // num_heads
    block_size = 16
    total_blocks = (batch_size * context_len) // block_size
    
    print(f"Allocating KV Cache for {total_blocks} blocks...")
    kv_caches = []
    for _ in range(hf_config.num_hidden_layers):
        k_cache = torch.zeros((total_blocks, block_size, num_kv_heads, head_size), device="cuda", dtype=torch.float16)
        v_cache = torch.zeros((total_blocks, block_size, num_kv_heads, head_size), device="cuda", dtype=torch.float16)
        kv_caches.append((k_cache, v_cache))

    attn_metadata = {
        "slot_mapping": torch.arange(batch_size, device="cuda", dtype=torch.int32),
        "seq_lens": torch.ones(batch_size, device="cuda", dtype=torch.int32) * context_len,
        "block_tables": torch.zeros((batch_size, total_blocks // batch_size), device="cuda", dtype=torch.int32)
    }

    # 4. Benchmark
    print("Executing stress test iterations...")
    iters = 10
    start_time = time.time()
    for _ in range(iters):
        with torch.inference_mode():
            model(input_ids, positions, kv_caches, attn_metadata)
    torch.cuda.synchronize()
    
    avg_latency = (time.time() - start_time) / iters * 1000
    tps = (1000 / avg_latency) * batch_size
    
    print(f"RESULT: Latency={avg_latency:.2f}ms, TPS={tps:.2f}")
    
    del model, kv_caches
    clear_gguf_cache()
    gc.collect()
    torch.cuda.empty_cache()
    
    return tps

if __name__ == "__main__":
    model_path = "models/Kimi-Linear-48B-GGUF"
    if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
        # We start with safe BS=1 and increase
        for bs in [1, 4, 8]:
            try:
                stress_test_kimi(model_path, batch_size=bs, context_len=1024)
            except Exception as e:
                print(f"BS={bs} failed: {e}")
                break
    else:
        print(f"Kimi path not found: {model_path}")
