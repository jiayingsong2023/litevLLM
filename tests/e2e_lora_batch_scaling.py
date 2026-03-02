# SPDX-License-Identifier: Apache-2.0
import torch
import time
from vllm.model_executor.model_loader import get_model

def benchmark_lora_batch(model_id, batch_size, rank=16):
    print(f"\n>>> Benchmarking LoRA: Batch Size = {batch_size}, Rank = {rank}")
    
    # 1. Setup Config
    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(model_id)
    
    class FakeVllmConfig:
        def __init__(self):
            self.model_config = type('obj', (object,), {
                'hf_config': hf_config,
                'dtype': torch.float16,
                'max_model_len': 2048,
                'model': model_id,
                'get_num_kv_heads': lambda x: hf_config.num_key_value_heads,
                'get_head_size': lambda: hf_config.hidden_size // hf_config.num_attention_heads,
                'get_num_layers': lambda x: hf_config.num_hidden_layers
            })
            self.parallel_config = type('obj', (object,), {})
            self.quant_config = None

    v_config = FakeVllmConfig()
    model = get_model(v_config).cuda().half()
    
    # 2. Inject LoRA Weights
    for module in model.modules():
        from vllm.model_executor.layers.lite_linear import LiteLinear
        if isinstance(module, LiteLinear):
            lora_a = torch.randn(rank, module.input_size, device="cuda", dtype=torch.float16)
            lora_b = torch.randn(module.output_size, rank, device="cuda", dtype=torch.float16)
            module.set_lora(lora_a, lora_b, scaling=2.0)

    # 3. Prepare Inputs
    input_ids = torch.ones((batch_size, 1), device="cuda", dtype=torch.long)
    positions = torch.zeros(batch_size, device="cuda", dtype=torch.long) + 10
    
    head_size = hf_config.hidden_size // hf_config.num_attention_heads
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
    
    avg_latency = (time.time() - start_time) / iters * 1000
    tps = (1000 / avg_latency) * batch_size
    
    print(f"RESULT: Latency={avg_latency:.2f}ms, Total Throughput={tps:.2f} tokens/sec")
    return tps

if __name__ == "__main__":
    mid = "models/TinyLlama-1.1B-Chat-v1.0"
    results = {}
    for bs in [1, 8, 32]:
        try:
            results[bs] = benchmark_lora_batch(mid, bs)
        except Exception as e:
            print(f"Batch Size {bs} failed: {e}")
    
    print("\n" + "="*40)
    print("LORA BATCH SCALING SUMMARY")
    print("="*40)
    for bs, tps in results.items():
        print(f"Batch Size {bs:2}: {tps:8.2f} tokens/sec")
    print("="*40)
