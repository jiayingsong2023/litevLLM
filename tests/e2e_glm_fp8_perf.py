import torch
import torch.nn as nn
import time
import os
import gc
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.layers.quantization.gguf import clear_gguf_cache

def benchmark_glm_fp8(model_path, batch_size, context_len=4096, enable_fp8=True):
    tag = "FP8" if enable_fp8 else "FP16"
    print(f"\n>>> Benchmarking GLM-4.7-Flash: BS = {batch_size}, Context = {context_len}, MODE = {tag}")
    
    # 环境配置
    os.environ["FASTINFERENCE_MOE_CACHE_MODE"] = "full"
    os.environ["FASTINFERENCE_DEEPSEEK_FP8"] = "1" if enable_fp8 else "0"
    os.environ["FASTINFERENCE_DEEPSEEK_GROUPED_MOE"] = "1"
    
    clear_gguf_cache()
    gc.collect()

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
            self.qk_nope_head_dim = 64 
            self.qk_rope_head_dim = 64
            self.v_head_dim = 128
            self.kv_lora_rank = 512
            self.q_lora_rank = 768
            self.first_k_dense_replace = 1
            self.architectures = ["DeepseekV2ForCausalLM"] # Reuse DeepSeek implementation
            self.dtype = "bfloat16"
    
    hf_config = GLM47FlashConfig()
    
    class FakeVllmConfig:
        def __init__(self):
            self.model_config = type('obj', (object,), {
                'hf_config': hf_config,
                'dtype': torch.float16,
                'max_model_len': context_len + 128,
                'model': model_path,
                'get_num_kv_heads': lambda x: hf_config.num_key_value_heads,
                'get_head_size': lambda: 192,
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
    model = get_model(v_config).cuda().half()
    
    input_ids = torch.ones((batch_size, 1), device="cuda", dtype=torch.long)
    positions = torch.zeros(batch_size, device="cuda", dtype=torch.long) + context_len
    
    num_blocks = (context_len // 16) * batch_size + 256
    kv_caches = []
    for _ in range(hf_config.num_hidden_layers):
        # GLM Flash MLA dims: key=192, value=128
        k_cache = torch.zeros((num_blocks, 16, hf_config.num_key_value_heads, 192), device="cuda", dtype=torch.float16)
        v_cache = torch.zeros((num_blocks, 16, hf_config.num_key_value_heads, 128), device="cuda", dtype=torch.float16)
        kv_caches.append((k_cache, v_cache))

    attn_metadata = {
        "slot_mapping": torch.arange(batch_size, device="cuda", dtype=torch.int32),
        "seq_lens": torch.ones(batch_size, device="cuda", dtype=torch.int32) * context_len,
        "block_tables": torch.arange(num_blocks, device="cuda", dtype=torch.int32).view(batch_size, -1)[:, :context_len//16]
    }

    # Warmup
    print(f"Warmup ({tag})...")
    for _ in range(2):
        with torch.inference_mode():
            model(input_ids, positions, kv_caches, attn_metadata)
    torch.cuda.synchronize()

    # Benchmark
    iters = 10
    print(f"Running {iters} iterations...")
    start_time = time.time()
    for _ in range(iters):
        with torch.inference_mode():
            model(input_ids, positions, kv_caches, attn_metadata)
    torch.cuda.synchronize()
    end_time = time.time()

    avg_latency = (end_time - start_time) / iters * 1000
    total_tps = (1000 / avg_latency) * batch_size
    
    print(f"RESULT ({tag}): Latency={avg_latency:.2f}ms, Total Throughput={total_tps:.2f} tokens/sec")
    
    del model, kv_caches
    clear_gguf_cache()
    gc.collect()
    torch.cuda.empty_cache()
    
    return total_tps

if __name__ == "__main__":
    model_path = "models/GLM-4.7-Flash-GGUF"
    
    # 降低长度以确保 FP16 在 60GB 显存内能跑起 Full Cache
    bs = 32
    cl = 1024
    
    try:
        tps16 = benchmark_glm_fp8(model_path, bs, context_len=cl, enable_fp8=False)
        tps8 = benchmark_glm_fp8(model_path, bs, context_len=cl, enable_fp8=True)
        
        print(f"\n>>> FINAL COMPARISON (GLM-4.7-Flash):")
        print(f"FP16: {tps16:.2f} TPS")
        print(f"FP8:  {tps8:.2f} TPS")
        print(f"Improvement: {((tps8/tps16) - 1)*100:.1f}%")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
