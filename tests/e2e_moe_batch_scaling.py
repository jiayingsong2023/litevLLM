# SPDX-License-Identifier: Apache-2.0
import torch
import time
from vllm.model_executor.model_loader import get_model

def benchmark_moe_batch(model_path, batch_size):
    print(f"\n>>> Benchmarking Qwen-MoE: Batch Size = {batch_size}")
    
    # 1. Setup Config
    class DummyHFConfig:
        def __init__(self):
            self.num_hidden_layers = 24
            self.num_attention_heads = 16
            self.num_key_value_heads = 16
            self.hidden_size = 2048
            self.intermediate_size = 5504
            self.max_position_embeddings = 2048
            self.vocab_size = 32000
            self.architectures = ["LlamaForCausalLM"]
            self.dtype = "float16"
            self.rms_norm_eps = 1e-6
    hf_config = DummyHFConfig()

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
    model = get_model(v_config).cuda().half()
    
    # 2. Prepare Inputs
    input_ids = torch.ones((batch_size, 1), device="cuda", dtype=torch.long)
    positions = torch.zeros(batch_size, device="cuda", dtype=torch.long) + 10
    
    kv_caches = []
    for _ in range(hf_config.num_hidden_layers):
        k_cache = torch.zeros((128, 16, hf_config.num_key_value_heads, 128), device="cuda", dtype=torch.float16)
        v_cache = torch.zeros((128, 16, hf_config.num_key_value_heads, 128), device="cuda", dtype=torch.float16)
        kv_caches.append((k_cache, v_cache))

    attn_metadata = {
        "slot_mapping": torch.arange(batch_size, device="cuda", dtype=torch.int32),
        "seq_lens": torch.ones(batch_size, device="cuda", dtype=torch.int32) * 11,
        "block_tables": torch.zeros((batch_size, 128), device="cuda", dtype=torch.int32)
    }

    # 3. Warmup
    for _ in range(5):
        with torch.inference_mode():
            model(input_ids, positions, kv_caches, attn_metadata)
    torch.cuda.synchronize()

    # 4. Benchmark
    iters = 20
    start_time = time.time()
    for _ in range(iters):
        with torch.inference_mode():
            model(input_ids, positions, kv_caches, attn_metadata)
    torch.cuda.synchronize()
    
    avg_latency = (time.time() - start_time) / iters * 1000
    tps = (1000 / avg_latency) * batch_size # 总吞吐量 = 每秒生成的总 token 数
    
    print(f"RESULT: Latency={avg_latency:.2f}ms, Total Throughput={tps:.2f} tokens/sec")
    return tps

if __name__ == "__main__":
    path = "models/Qwen1.5-MoE-A2.7B-Chat.Q8_0.gguf"
    results = {}
    for bs in [1, 8, 32]:
        results[bs] = benchmark_moe_batch(path, bs)
    
    print("\n" + "="*40)
    print("QWEN-MOE BATCH SCALING SUMMARY")
    print("="*40)
    for bs, tps in results.items():
        print(f"Batch Size {bs:2}: {tps:8.2f} tokens/sec")
    print("="*40)
