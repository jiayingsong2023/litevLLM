# SPDX-License-Identifier: Apache-2.0
import torch
import time
import os
import json
import gc
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.layers.quantization.gguf import clear_gguf_cache

def benchmark_qwen3_5(model_path, batch_size=32):
    print(f"\n>>> Benchmarking Qwen3.5: {model_path} (Batch Size {batch_size})")
    
    # Pre-run cleanup
    clear_gguf_cache()
    gc.collect()
    
    # 1. Load config
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        config_data = json.load(f)
    
    text_config_data = config_data.get("text_config", config_data)
    
    class DummyHFConfig:
        def __init__(self, data):
            self.num_hidden_layers = data.get("num_hidden_layers", 32)
            self.num_attention_heads = data.get("num_attention_heads", 16)
            self.num_key_value_heads = data.get("num_key_value_heads", 4)
            self.hidden_size = data.get("hidden_size", 4096)
            self.intermediate_size = data.get("intermediate_size", 12288)
            self.max_position_embeddings = data.get("max_position_embeddings", 262144)
            self.vocab_size = data.get("vocab_size", 248320)
            self.rms_norm_eps = data.get("rms_norm_eps", 1e-6)
            self.architectures = config_data.get("architectures", ["Qwen3_5ForConditionalGeneration"])
            self.dtype = data.get("torch_dtype", "bfloat16")
            # MoE specific
            self.num_experts = data.get("num_experts", 0)
            self.num_experts_per_tok = data.get("num_experts_per_tok", 0)
            self.moe_intermediate_size = data.get("moe_intermediate_size", 0)
            self.first_k_dense_replace = data.get("first_k_dense_replace", 0)

    hf_config = DummyHFConfig(text_config_data)

    class FakeVllmConfig:
        def __init__(self):
            self.model_config = type('obj', (object,), {
                'hf_config': hf_config,
                'dtype': torch.float16,
                'max_model_len': 2048,
                'model': model_path,
                'get_num_kv_heads': lambda x: hf_config.num_key_value_heads,
                'get_head_size': lambda: hf_config.hidden_size // hf_config.num_attention_heads,
                'get_num_layers': lambda x: hf_config.num_hidden_layers
            })
            self.parallel_config = type('obj', (object,), {})
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
    # Use standard cache size for 9B, reduced for MoE if needed
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
    
    # Explicit cleanup
    del model
    clear_gguf_cache()
    gc.collect()
    
    return total_tps

if __name__ == "__main__":
    results = {}
    
    # Test Qwen3.5-9B (Dense)
    model_9b = "models/Qwen3.5-9B-GGUF"
    if os.path.exists(model_9b) and os.path.exists(os.path.join(model_9b, "config.json")):
        try:
            results["Qwen3.5-9B"] = benchmark_qwen3_5(model_9b, batch_size=32)
        except Exception as e:
            print(f"Qwen3.5-9B failed: {e}")

    # Test Qwen3.5-35B (MoE)
    model_35b = "models/Qwen3.5-35B-MoE-GGUF"
    if os.path.exists(model_35b) and os.path.exists(os.path.join(model_35b, "config.json")):
        gguf_files = [f for f in os.listdir(model_35b) if f.endswith(".gguf")]
        if gguf_files:
            weights_35b = os.path.join(model_35b, gguf_files[0])
            try:
                if os.path.getsize(weights_35b) > 10 * 1024 * 1024 * 1024:
                    # Stress test BS=16 and BS=32
                    for bs in [16, 32]:
                        results[f"Qwen3.5-35B-MoE-BS{bs}"] = benchmark_qwen3_5(model_35b, batch_size=bs)
                else:
                    print(f"Qwen3.5-35B weight file {gguf_files[0]} is too small, skipping...")
            except Exception as e:
                print(f"Qwen3.5-35B-MoE failed: {e}")
                import traceback
                traceback.print_exc()
    print("\n" + "="*45)
    print("QWEN3.5 PERFORMANCE SUMMARY")
    print("="*45)
    for name, tps in results.items():
        print(f"{name:25}: {tps:8.2f} tokens/sec")
    print("="*45)
