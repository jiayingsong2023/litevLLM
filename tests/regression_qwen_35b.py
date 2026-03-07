import torch
import torch.nn as nn
import time
import os
import gc
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.layers.quantization.gguf import clear_gguf_cache

def benchmark_qwen_35b(model_path, batch_size, context_len=1024):
    print(f"\n>>> [Regression] Qwen-3.5-35B: BS={batch_size}, Context={context_len}, FP8")
    os.environ["FASTINFERENCE_DEEPSEEK_FP8"] = "1"
    clear_gguf_cache(); gc.collect()

    class QwenMoeConfig:
        def __init__(self):
            self.num_hidden_layers = 40; self.num_attention_heads = 16; self.num_key_value_heads = 2
            self.hidden_size = 2048; self.intermediate_size = 512; self.max_position_embeddings = 262144
            self.vocab_size = 248320; self.rms_norm_eps = 1e-6; self.num_experts = 256
            self.num_experts_per_tok = 8; self.architectures = ["Qwen2ForCausalLM"]
            self.dtype = "bfloat16"
    
    hf_config = QwenMoeConfig()
    class FakeVllmConfig:
        def __init__(self):
            self.model_config = type('obj', (object,), {
                'hf_config': hf_config, 'dtype': torch.float16, 'max_model_len': context_len + 128,
                'model': model_path, 'get_num_kv_heads': lambda x: hf_config.num_key_value_heads,
                'get_head_size': lambda: 256, 'get_num_layers': lambda x: hf_config.num_hidden_layers,
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
    kv_caches = [(torch.zeros((num_blocks, 16, hf_config.num_key_value_heads, 256), device="cuda", dtype=torch.float16),
                  torch.zeros((num_blocks, 16, hf_config.num_key_value_heads, 256), device="cuda", dtype=torch.float16))
                 for _ in range(hf_config.num_hidden_layers)]

    attn_metadata = {
        "slot_mapping": torch.arange(batch_size, device="cuda", dtype=torch.int32),
        "seq_lens": torch.ones(batch_size, device="cuda", dtype=torch.int32) * context_len,
        "block_tables": torch.arange(num_blocks, device="cuda", dtype=torch.int32).view(batch_size, -1)[:, :context_len//16]
    }

    for _ in range(1): 
        with torch.inference_mode(): model(input_ids, positions, kv_caches, attn_metadata)
    torch.cuda.synchronize()

    iters = 5; start_time = time.time()
    for _ in range(iters):
        with torch.inference_mode(): model(input_ids, positions, kv_caches, attn_metadata)
    torch.cuda.synchronize()
    
    latency = (time.time() - start_time) / iters * 1000
    print(f"RESULT: Latency={latency:.2f}ms, TPS={batch_size*1000/latency:.2f}")
    del model, kv_caches; clear_gguf_cache(); gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__":
    for bs in [1, 2, 8]:
        try:
            benchmark_qwen_35b("models/Qwen3.5-35B-MoE-GGUF", bs, 1024)
        except Exception as e:
            print(f"BS={bs} failed: {e}")
