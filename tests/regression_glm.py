import torch
import torch.nn as nn
import time
import os
import gc
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.layers.quantization.gguf import clear_gguf_cache

def benchmark_glm_fp8(model_path, batch_size, context_len=4096):
    # Balanced mode settings
    os.environ["FASTINFERENCE_MOE_CACHE_MODE"] = "dynamic"
    os.environ["FASTINFERENCE_MOE_LRU_SIZE"] = "32" # Recover to 32 for better hits
    
    print(f"\n>>> [Regression] GLM-4.7-Flash: BS={batch_size}, Context={context_len}, FP8 (Balanced Mode)")
    os.environ["FASTINFERENCE_DEEPSEEK_FP8"] = "1"
    os.environ["FASTINFERENCE_DEEPSEEK_GROUPED_MOE"] = "1"
    
    clear_gguf_cache(); gc.collect()

    class GLM47FlashConfig:
        def __init__(self):
            self.num_hidden_layers = 28; self.num_attention_heads = 40; self.num_key_value_heads = 20
            self.hidden_size = 2048; self.intermediate_size = 10944; self.max_position_embeddings = 131072
            self.vocab_size = 154880; self.rms_norm_eps = 1e-6; self.n_routed_experts = 64
            self.num_experts_per_tok = 6; self.moe_intermediate_size = 1536; self.qk_nope_head_dim = 64 
            self.qk_rope_head_dim = 64; self.v_head_dim = 128; self.kv_lora_rank = 512
            self.q_lora_rank = 768; self.first_k_dense_replace = 1; self.architectures = ["DeepseekV2ForCausalLM"]
            self.dtype = "bfloat16"
    
    hf_config = GLM47FlashConfig()
    class FakeVllmConfig:
        def __init__(self):
            self.model_config = type('obj', (object,), {
                'hf_config': hf_config, 'dtype': torch.float16, 'max_model_len': context_len + 128,
                'model': model_path, 'get_num_kv_heads': lambda x: hf_config.num_key_value_heads,
                'get_head_size': lambda: 192, 'get_num_layers': lambda x: hf_config.num_hidden_layers,
                'get_total_num_kv_heads': lambda: hf_config.num_key_value_heads, 'get_max_model_len': lambda: context_len + 128,
            })
            self.parallel_config = type('obj', (object,), {'tensor_parallel_size': 1, 'pipeline_parallel_size': 1, 'world_size': 1})
            from vllm.model_executor.layers.quantization.gguf import GGUFConfig
            self.quant_config = GGUFConfig()

    v_config = FakeVllmConfig()
    model = get_model(v_config).cuda().half()
    
    input_ids = torch.ones((batch_size, 1), device="cuda", dtype=torch.long)
    positions = torch.zeros(batch_size, device="cuda", dtype=torch.long) + context_len
    num_blocks = (context_len // 16) * batch_size + 256
    kv_caches = [(torch.zeros((num_blocks, 16, hf_config.num_key_value_heads, 192), device="cuda", dtype=torch.float16),
                  torch.zeros((num_blocks, 16, hf_config.num_key_value_heads, 128), device="cuda", dtype=torch.float16))
                 for _ in range(hf_config.num_hidden_layers)]

    attn_metadata = {
        "slot_mapping": torch.arange(batch_size, device="cuda", dtype=torch.int32),
        "seq_lens": torch.ones(batch_size, device="cuda", dtype=torch.int32) * context_len,
        "block_tables": torch.arange(num_blocks, device="cuda", dtype=torch.int32).view(batch_size, -1)[:, :context_len//16]
    }

    print(f"Warmup (Balanced Mode)...")
    for _ in range(1): 
        with torch.inference_mode(): model(input_ids, positions, kv_caches, attn_metadata)
    torch.cuda.synchronize()

    iters = 10; start_time = time.time()
    for _ in range(iters):
        with torch.inference_mode(): model(input_ids, positions, kv_caches, attn_metadata)
    torch.cuda.synchronize()
    
    latency = (time.time() - start_time) / iters * 1000
    print(f"RESULT: Latency={latency:.2f}ms, TPS={batch_size*1000/latency:.2f}")
    del model, kv_caches; clear_gguf_cache(); gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__":
    benchmark_glm_fp8("models/GLM-4.7-Flash-GGUF", 32, 4096)
