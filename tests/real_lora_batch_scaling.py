# SPDX-License-Identifier: Apache-2.0
import torch
import time
import os
from vllm import LLM, SamplingParams

def run_lora_batch_scaling(batch_size=32, rank=16):
    print(f"=== TRUE BATCH LORA PERFORMANCE: BS={batch_size}, Rank={rank} ===")
    model_path = "models/TinyLlama-1.1B-Chat-v1.0"
    
    # 1. Load Model
    llm = LLM(model=model_path)
    model = llm.model
    
    # 2. Inject LoRA into all layers
    print(f">>> Injecting Rank-{rank} Adapters...")
    for name, module in model.named_modules():
        from vllm.model_executor.layers.lite_linear import LiteLinear
        if isinstance(module, LiteLinear):
            la = torch.randn(rank, module.input_size, device="cuda", dtype=torch.float16)
            lb = torch.randn(module.output_size, rank, device="cuda", dtype=torch.float16)
            module.add_adapter(aid=101, lora_a=la, lora_b=lb, scaling=2.0)

    # 3. Setup Batch Input
    lora_mapping = torch.full((batch_size,), 101, device="cuda", dtype=torch.int32)
    
    # 4. Prepare Metadata
    input_ids = torch.ones((batch_size, 1), device="cuda", dtype=torch.long)
    positions = torch.zeros((batch_size, 1), device="cuda", dtype=torch.long)
    num_kv_heads = llm.model_cfg.get_num_kv_heads(None)
    head_size = llm.model_cfg.get_head_size()
    num_layers = llm.model_cfg.get_num_layers(None)
    kv_caches = [(torch.zeros((128, 16, num_kv_heads, head_size), device="cuda", dtype=torch.float16),
                  torch.zeros((128, 16, num_kv_heads, head_size), device="cuda", dtype=torch.float16)) for _ in range(num_layers)]
    attn_meta = {"slot_mapping": torch.arange(batch_size, device="cuda", dtype=torch.int32), 
                 "seq_lens": torch.ones(batch_size, device="cuda", dtype=torch.int32)}
    
    # 5. Warmup
    print("Warmup...")
    for _ in range(5):
        with torch.inference_mode():
            _ = model(input_ids, positions, kv_caches, attn_meta, lora_mapping=lora_mapping)
    torch.cuda.synchronize()

    # 6. Benchmark
    iters = 50
    print(f"Benchmarking {iters} iterations...")
    start = time.time()
    for _ in range(iters):
        with torch.inference_mode():
            _ = model(input_ids, positions, kv_caches, attn_meta, lora_mapping=lora_mapping)
    torch.cuda.synchronize()
    end = time.time()
    
    latency = (end - start) / iters * 1000
    tps = (1000 / latency) * batch_size
    
    print(f"\n======================================")
    print(f"LORA BS={batch_size} RESULT")
    print(f"Throughput: {tps:.2f} tokens/sec")
    print(f"Latency:    {latency:.2f} ms")
    print(f"======================================")
    return tps

if __name__ == "__main__":
    run_lora_batch_scaling(batch_size=32)
