# SPDX-License-Identifier: Apache-2.0
import torch
import time
import os
import json
import gc
from vllm.model_executor.model_loader import get_model
from transformers import AutoConfig
from vllm.model_executor.layers.quantization.gguf import clear_gguf_cache

def load_real_config(model_path):
    """Loads configuration from a real model directory."""
    if os.path.isdir(model_path):
        config_file = os.path.join(model_path, "config.json")
        if os.path.exists(config_file):
            # For Qwen3.5 and others with nested configs
            with open(config_file, "r") as f:
                data = json.load(f)
            
            # Simple wrapper to mimic HF config object
            class RealConfig:
                def __init__(self, d, architectures):
                    self.__dict__.update(d)
                    self.architectures = architectures
            
            text_config = data.get("text_config", data)
            archs = data.get("architectures", [])
            return RealConfig(text_config, archs)
            
    # Fallback to transformers for standard models
    return AutoConfig.from_pretrained(model_path, trust_remote_code=True)

def benchmark_real_model(name, model_path, batch_size=32, context_len=11):
    print(f"\n>>> [REAL] Benchmarking: {name}")
    print(f">>> Path: {model_path}, BS: {batch_size}, Ctx: {context_len}")
    
    clear_gguf_cache()
    gc.collect()
    
    # 1. Load Real Config
    hf_config = load_real_config(model_path)
    
    class FakeVllmConfig:
        def __init__(self):
            # Extract attributes safely
            n_kv_heads = getattr(hf_config, "num_key_value_heads", getattr(hf_config, "num_attention_heads", 1))
            h_size = getattr(hf_config, "hidden_size", 4096)
            n_heads = getattr(hf_config, "num_attention_heads", 32)
            n_layers = getattr(hf_config, "num_hidden_layers", 32)
            
            self.model_config = type('obj', (object,), {
                'hf_config': hf_config,
                'dtype': torch.float16,
                'max_model_len': 4096,
                'model': model_path,
                'get_num_kv_heads': lambda x: n_kv_heads,
                'get_head_size': lambda: h_size // n_heads,
                'get_num_layers': lambda x: n_layers,
                'get_total_num_kv_heads': lambda: n_kv_heads,
                'get_max_model_len': lambda: 4096,
            })
            self.parallel_config = type('obj', (object,), {
                'tensor_parallel_size': 1,
                'pipeline_parallel_size': 1,
                'world_size': 1,
            })
            # Auto-detect quantization
            self.quant_config = None
            if any(f.endswith(".gguf") for f in (os.listdir(model_path) if os.path.isdir(model_path) else [])):
                from vllm.model_executor.layers.quantization.gguf import GGUFConfig
                self.quant_config = GGUFConfig()

    v_config = FakeVllmConfig()
    
    # 2. Load Model & Weights
    print("Loading weights...")
    model = get_model(v_config).cuda().half()
    
    # 3. Prepare Inputs
    input_ids = torch.ones((batch_size, 1), device="cuda", dtype=torch.long)
    positions = torch.zeros(batch_size, device="cuda", dtype=torch.long) + (context_len - 1)
    
    num_heads = hf_config.num_attention_heads
    num_kv_heads = getattr(hf_config, "num_key_value_heads", num_heads)
    head_size = hf_config.hidden_size // num_heads
    
    kv_caches = []
    num_blocks = 256 # Safe default
    for _ in range(hf_config.num_hidden_layers):
        k_cache = torch.zeros((num_blocks, 16, num_kv_heads, head_size), device="cuda", dtype=torch.float16)
        v_cache = torch.zeros((num_blocks, 16, num_kv_heads, head_size), device="cuda", dtype=torch.float16)
        kv_caches.append((k_cache, v_cache))

    attn_metadata = {
        "slot_mapping": torch.arange(batch_size, device="cuda", dtype=torch.int32),
        "seq_lens": torch.ones(batch_size, device="cuda", dtype=torch.int32) * context_len,
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
    start_time = time.time()
    for _ in range(iters):
        with torch.inference_mode():
            model(input_ids, positions, kv_caches, attn_metadata)
    torch.cuda.synchronize()
    end_time = time.time()

    avg_latency = (end_time - start_time) / iters * 1000
    tps = (1000 / avg_latency) * batch_size
    
    print(f"RESULT: Latency={avg_latency:.2f}ms, Throughput={tps:.2f} tokens/sec")
    
    del model, kv_caches
    clear_gguf_cache()
    gc.collect()
    return tps

if __name__ == "__main__":
    test_matrix = [
        ("TinyLlama-1.1B", "models/TinyLlama-1.1B-Chat-v1.0", 32),
        ("Qwen3.5-9B-GGUF", "models/Qwen3.5-9B-GGUF", 32),
        ("GLM-4.7-Flash-GGUF", "models/GLM-4.7-Flash-GGUF", 32),
        ("DeepSeek-V2-Lite-GGUF", "models/DeepSeek-V2-Lite-GGUF", 32),
        ("Kimi-Linear-48B-GGUF", "models/Kimi-Linear-48B-GGUF", 4),
        ("Qwen3.5-35B-MoE-GGUF", "models/Qwen3.5-35B-MoE-GGUF", 32),
    ]
    
    results = {}
    for name, path, bs in test_matrix:
        if os.path.exists(path):
            try:
                results[name] = benchmark_real_model(name, path, batch_size=bs)
            except Exception as e:
                print(f"Failed to benchmark {name}: {e}")
                # import traceback
                # traceback.print_exc()
        else:
            print(f"Skipping {name}, path not found: {path}")

    print("\n" + "="*50)
    print("FINAL REAL-WEIGHT PERFORMANCE SUMMARY")
    print("="*50)
    for name, tps in results.items():
        print(f"{name:25}: {tps:8.2f} tokens/sec")
    print("="*50)
