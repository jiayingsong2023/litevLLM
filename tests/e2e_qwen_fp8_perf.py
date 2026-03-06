import torch
import torch.nn as nn
import time
import os
import gc
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.layers.quantization.gguf import clear_gguf_cache

def benchmark_qwen_fp8(model_path, batch_size, context_len=1024):
    print(f"\n>>> Benchmarking Qwen-9B: BS = {batch_size}, Context = {context_len}, MODE = FP8")
    
    # 开启关键优化环境变量
    os.environ["FASTINFERENCE_DEEPSEEK_FP8"] = "1"
    
    clear_gguf_cache()
    gc.collect()

    class Qwen2Config:
        def __init__(self):
            self.num_hidden_layers = 28
            self.num_attention_heads = 28
            self.num_key_value_heads = 4
            self.hidden_size = 3584
            self.intermediate_size = 18944
            self.max_position_embeddings = 32768
            self.vocab_size = 152064
            self.rms_norm_eps = 1e-6
            self.architectures = ["Qwen2ForCausalLM"]
            self.dtype = "bfloat16"
    
    hf_config = Qwen2Config()
    
    class FakeVllmConfig:
        def __init__(self):
            self.model_config = type('obj', (object,), {
                'hf_config': hf_config,
                'dtype': torch.float16,
                'max_model_len': context_len + 128,
                'model': model_path,
                'get_num_kv_heads': lambda x: hf_config.num_key_value_heads,
                'get_head_size': lambda: 128,
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
    
    # Qwen2ForCausalLM will use Qwen2Model -> LlamaModel -> LiteDecoderLayer
    # LiteDecoderLayer uses LiteLinear which we just upgraded.
    model = get_model(v_config).cuda().half()
    
    input_ids = torch.ones((batch_size, 1), device="cuda", dtype=torch.long)
    positions = torch.zeros(batch_size, device="cuda", dtype=torch.long) + context_len
    
    num_blocks = (context_len // 16) * batch_size + 256
    kv_caches = []
    for _ in range(hf_config.num_hidden_layers):
        # Qwen-9B uses GQA, 4 KV heads, head_size 128
        k_cache = torch.zeros((num_blocks, 16, hf_config.num_key_value_heads, 128), device="cuda", dtype=torch.float16)
        v_cache = torch.zeros((num_blocks, 16, hf_config.num_key_value_heads, 128), device="cuda", dtype=torch.float16)
        kv_caches.append((k_cache, v_cache))

    attn_metadata = {
        "slot_mapping": torch.arange(batch_size, device="cuda", dtype=torch.int32),
        "seq_lens": torch.ones(batch_size, device="cuda", dtype=torch.int32) * context_len,
        "block_tables": torch.arange(num_blocks, device="cuda", dtype=torch.int32).view(batch_size, -1)[:, :context_len//16]
    }

    # Warmup + FP8 Conversion
    print("Warmup + FP8 Conversion...")
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
    
    print(f"RESULT: Latency={avg_latency:.2f}ms, Total Throughput={total_tps:.2f} tokens/sec")
    
    del model, kv_caches
    clear_gguf_cache()
    gc.collect()
    torch.cuda.empty_cache()
    
    return total_tps

if __name__ == "__main__":
    # 使用 Qwen3.5-9B-GGUF 模型路径
    model_path = "models/Qwen3.5-9B-GGUF"
    
    # 测试 Batch Size = 32, Context = 1024
    try:
        benchmark_qwen_fp8(model_path, 32, context_len=1024)
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
