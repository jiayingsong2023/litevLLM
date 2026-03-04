# SPDX-License-Identifier: Apache-2.0
import torch
import time
import os
import json
import gc
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.layers.quantization.gguf import clear_gguf_cache

def benchmark_qwen3_5_long(model_path, batch_size=32, context_len=4096):
    print(f"\n>>> Benchmarking Qwen3.5 LONG CTX: {model_path}")
    print(f">>> Context Length: {context_len}, Batch Size: {batch_size}")
    
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
            self.num_experts = data.get("num_experts", 0)
            self.num_experts_per_tok = data.get("num_experts_per_tok", 0)
            self.moe_intermediate_size = data.get("moe_intermediate_size", 0)

    hf_config = DummyHFConfig(text_config_data)

    class FakeVllmConfig:
        def __init__(self):
            self.model_config = type('obj', (object,), {
                'hf_config': hf_config,
                'dtype': torch.float16,
                'max_model_len': context_len + 128,
                'model': model_path,
                'get_num_kv_heads': lambda x: hf_config.num_key_value_heads,
                'get_head_size': lambda: hf_config.hidden_size // hf_config.num_attention_heads,
                'get_num_layers': lambda x: hf_config.num_hidden_layers
            })
            self.parallel_config = type('obj', (object,), {})
            from vllm.model_executor.layers.quantization.gguf import GGUFConfig
            self.quant_config = GGUFConfig()

    v_config = FakeVllmConfig()
    model = get_model(v_config).cuda().half()
    
    # 2. Prepare Long Context Inputs
    input_ids = torch.ones((batch_size, 1), device="cuda", dtype=torch.long)
    positions = torch.zeros(batch_size, device="cuda", dtype=torch.long) + (context_len - 1)
    
    num_heads = hf_config.num_attention_heads
    num_kv_heads = hf_config.num_key_value_heads
    head_size = hf_config.hidden_size // num_heads
    block_size = 16
    blocks_per_req = context_len // block_size
    total_blocks = batch_size * blocks_per_req
    
    print(f"Allocating KV Cache for {total_blocks} blocks...")
    kv_caches = []
    for _ in range(hf_config.num_hidden_layers):
        k_cache = torch.zeros((total_blocks, block_size, num_kv_heads, head_size), device="cuda", dtype=torch.float16)
        v_cache = torch.zeros((total_blocks, block_size, num_kv_heads, head_size), device="cuda", dtype=torch.float16)
        kv_caches.append((k_cache, v_cache))

    block_tables = torch.arange(total_blocks, device="cuda", dtype=torch.int32).view(batch_size, blocks_per_req)
    
    attn_metadata = {
        "slot_mapping": torch.arange(batch_size, device="cuda", dtype=torch.int32) * blocks_per_req + (blocks_per_req - 1),
        "seq_lens": torch.ones(batch_size, device="cuda", dtype=torch.int32) * context_len,
        "block_tables": block_tables
    }

    # 3. Warmup
    print("Warmup...")
    for _ in range(3):
        with torch.inference_mode():
            model(input_ids, positions, kv_caches, attn_metadata)
    torch.cuda.synchronize()

    # 4. Benchmark
    iters = 20
    print(f"Running {iters} iterations at 4K context...")
    start_time = time.time()
    for _ in range(iters):
        with torch.inference_mode():
            model(input_ids, positions, kv_caches, attn_metadata)
    torch.cuda.synchronize()
    
    avg_latency = (time.time() - start_time) / iters * 1000
    total_tps = (1000 / avg_latency) * batch_size
    
    print(f"RESULT (4K CTX): Latency={avg_latency:.2f}ms, Throughput={total_tps:.2f} tokens/sec")
    
    del model, kv_caches
    clear_gguf_cache()
    gc.collect()
    return total_tps

if __name__ == "__main__":
    model_9b = "models/Qwen3.5-9B-GGUF"
    model_35b = "models/Qwen3.5-35B-MoE-GGUF"
    results = {}
    
    try:
        results["Qwen3.5-9B-BS32-4K"] = benchmark_qwen3_5_long(model_9b, batch_size=32, context_len=4096)
    except Exception as e:
        print(f"9B 4K failed: {e}")

    try:
        results["Qwen3.5-35B-MoE-BS16-4K"] = benchmark_qwen3_5_long(model_35b, batch_size=16, context_len=4096)
        results["Qwen3.5-35B-MoE-BS32-4K"] = benchmark_qwen3_5_long(model_35b, batch_size=32, context_len=4096)
    except Exception as e:
        print(f"35B 4K failed: {e}")

    print("\n" + "="*50)
    print("QWEN3.5 LONG CONTEXT (4K) PERFORMANCE SUMMARY")
    print("="*50)
    for name, tps in results.items():
        print(f"{name:30}: {tps:8.2f} tokens/sec")
    print("="*50)
