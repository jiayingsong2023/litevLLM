# SPDX-License-Identifier: Apache-2.0
import torch
import time
import os
from vllm.model_executor.model_loader import get_model
from transformers import AutoConfig

def benchmark_model(model_path, is_gguf=False, is_moe=False):
    print(f"\n>>> Benchmarking: {model_path} (GGUF: {is_gguf}, MoE: {is_moe})")
    
    # 1. Setup Config
    if is_gguf:
        class DummyHFConfig:
            def __init__(self):
                self.num_hidden_layers = 32 if "7b" in model_path.lower() else 24
                self.num_attention_heads = 32
                self.num_key_value_heads = 32
                self.hidden_size = 4096 if "7b" in model_path.lower() else 2048
                self.intermediate_size = 11008 if "7b" in model_path.lower() else 5504
                self.max_position_embeddings = 2048
                self.vocab_size = 32000
                self.architectures = ["LlamaForCausalLM"]
                self.dtype = "float16"
                self.rms_norm_eps = 1e-6
        hf_config = DummyHFConfig()
    else:
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

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
            self.quant_config = None
            self.cache_config = type('obj', (object,), {'num_gpu_blocks': 128, 'block_size': 16})
            self.scheduler_config = type('obj', (object,), {})
            if is_gguf:
                from vllm.model_executor.layers.quantization.gguf import GGUFConfig
                self.quant_config = GGUFConfig()

    v_config = FakeVllmConfig()
    
    # 2. Load Model
    print("Initializing model...")
    model = get_model(v_config).cuda().half()
    
    # 3. Prepare Inputs (Batch=1, Decode)
    input_ids = torch.tensor([[1]], device="cuda")
    positions = torch.tensor([10], device="cuda")
    
    num_heads = hf_config.num_attention_heads
    num_kv_heads = hf_config.num_key_value_heads
    head_size = hf_config.hidden_size // num_heads
    
    kv_caches = []
    # Match layout: [num_blocks, block_size, num_kv_heads, head_dim]
    for _ in range(hf_config.num_hidden_layers):
        k_cache = torch.zeros((128, 16, num_kv_heads, head_size), device="cuda", dtype=torch.float16)
        v_cache = torch.zeros((128, 16, num_kv_heads, head_size), device="cuda", dtype=torch.float16)
        kv_caches.append((k_cache, v_cache))

    # Updated metadata format
    attn_metadata = {
        "slot_mapping": torch.tensor([10], device="cuda", dtype=torch.int32),
        "seq_lens": torch.tensor([11], device="cuda", dtype=torch.int32),
        "block_tables": torch.zeros((1, 128), device="cuda", dtype=torch.int32)
    }

    # 4. Warmup
    print("Warmup...")
    for _ in range(5):
        with torch.inference_mode():
            model(input_ids, positions, kv_caches, attn_metadata)
    torch.cuda.synchronize()

    # 5. Benchmark
    iters = 100
    start_time = time.time()
    for _ in range(iters):
        with torch.inference_mode():
            model(input_ids, positions, kv_caches, attn_metadata)
    torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    avg_latency = (total_time / iters) * 1000
    tps = 1000 / avg_latency
    
    print(f"RESULT: Latency={avg_latency:.2f}ms, Throughput={tps:.2f} tokens/sec")
    return tps

if __name__ == "__main__":
    results = {}
    
    # 1. TinyLlama
    try:
        results["TinyLlama"] = benchmark_model("models/TinyLlama-1.1B-Chat-v1.0")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"TinyLlama failed: {e}")

    # 2. Llama-2-7B GGUF
    try:
        results["Llama-7B-GGUF"] = benchmark_model("llama-2-7b-chat.Q4_K_M.gguf", is_gguf=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Llama-7B-GGUF failed: {e}")

    # 3. Qwen-MoE GGUF
    try:
        results["Qwen-MoE"] = benchmark_model("models/Qwen1.5-MoE-A2.7B-Chat.Q8_0.gguf", is_gguf=True, is_moe=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Qwen-MoE failed: {e}")

    print("\n" + "="*30)
    print("FINAL E2E PERFORMANCE SUMMARY")
    print("="*30)
    for name, tps in results.items():
        print(f"{name:15}: {tps:8.2f} tokens/sec")
    print("="*30)
