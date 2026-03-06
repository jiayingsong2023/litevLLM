# SPDX-License-Identifier: Apache-2.0
import torch
import time
import os
import gc
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.layers.quantization.gguf import clear_gguf_cache

def benchmark_glm_4_7_flash(model_path, batch_size):
    print(f"\n>>> Benchmarking Real GLM-4.7-Flash: Batch Size = {batch_size}")
    
    clear_gguf_cache()
    gc.collect()

    # 1. Config based on GLM-4.7-Flash GGUF metadata
    class GLM47FlashConfig:
        def __init__(self):
            self.num_hidden_layers = 28
            self.num_attention_heads = 40 
            self.num_key_value_heads = 20
            self.hidden_size = 2048
            self.intermediate_size = 10944
            self.max_position_embeddings = 131072
            self.vocab_size = 154880
            self.rms_norm_eps = 1e-6
            self.n_routed_experts = 64
            self.num_experts_per_tok = 6
            self.moe_intermediate_size = 1536
            self.qk_nope_head_dim = 64 # Corrected: 64 + 64 = 128
            self.qk_rope_head_dim = 64
            self.v_head_dim = 128
            self.kv_lora_rank = 512
            self.q_lora_rank = 768
            self.first_k_dense_replace = 1
            self.architectures = ["DeepseekV2ForCausalLM"]
            self.dtype = "bfloat16"
    
    hf_config = GLM47FlashConfig()
    
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
    print("Loading model weights (GGUF + Optimized Paths)...")
    model = get_model(v_config).cuda().half()
    
    # 3. Prepare Inputs
    input_ids = torch.ones((batch_size, 1), device="cuda", dtype=torch.long)
    positions = torch.zeros(batch_size, device="cuda", dtype=torch.long) + 10
    
    head_size = 128 # Standard for these models
    kv_caches = []
    for _ in range(hf_config.num_hidden_layers):
        k_cache = torch.zeros((128, 16, hf_config.num_key_value_heads, head_size), device="cuda", dtype=torch.float16)
        v_cache = torch.zeros((128, 16, hf_config.num_key_value_heads, head_size), device="cuda", dtype=torch.float16)
        kv_caches.append((k_cache, v_cache))

    attn_metadata = {
        "slot_mapping": torch.arange(batch_size, device="cuda", dtype=torch.int32),
        "seq_lens": torch.ones(batch_size, device="cuda", dtype=torch.int32) * 11,
        "block_tables": torch.zeros((batch_size, 128), device="cuda", dtype=torch.int32)
    }

    # 4. Warmup
    print("Warmup...")
    for _ in range(3):
        with torch.inference_mode():
            model(input_ids, positions, kv_caches, attn_metadata)
    torch.cuda.synchronize()

    # 5. Benchmark
    iters = 15
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
    
    # 开启平衡模式 (Dynamic LRU Cache)
    os.environ["FASTINFERENCE_MOE_CACHE_MODE"] = "dynamic"
    os.environ["FASTINFERENCE_MOE_LRU_SIZE"] = "16" 
    os.environ["FASTINFERENCE_DEEPSEEK_GROUPED_MOE"] = "1"
    
    # 仅测试 BS=32
    for bs in [32]:
        try:
            benchmark_glm_4_7_flash(model_path, bs)
        except Exception as e:
            print(f"Batch Size {bs} failed: {e}")
            import traceback
            traceback.print_exc()
