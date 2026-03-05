# SPDX-License-Identifier: Apache-2.0
import torch
import time
import os
from vllm import LLM, SamplingParams

def run_moe_lora_benchmark(batch_size=32, rank=16):
    print(f"=== MOE + LORA PERFORMANCE AUDIT: BS={batch_size}, Rank={rank} ===")
    model_path = "models/DeepSeek-V2-Lite-GGUF"
    
    llm = LLM(model=model_path)
    model = llm.model
    
    print(f">>> Injecting Rank-{rank} Adapters into MLA layers...")
    inject_count = 0
    for name, module in model.named_modules():
        from vllm.model_executor.layers.lite_linear import LiteLinear
        if isinstance(module, LiteLinear) and "self_attn" in name:
            la = torch.randn(rank, module.input_size, device="cuda", dtype=torch.float16)
            lb = torch.randn(module.output_size, rank, device="cuda", dtype=torch.float16)
            module.add_adapter(aid=101, lora_a=la, lora_b=lb, scaling=2.0)
            inject_count += 1
    print(f">>> Successfully injected {inject_count} LoRA adapters.")

    lora_mapping = torch.full((batch_size,), 101, device="cuda", dtype=torch.int32)
    
    input_ids = torch.ones((batch_size, 1), device="cuda", dtype=torch.long)
    positions = torch.zeros((batch_size, 1), device="cuda", dtype=torch.long)
    num_kv_heads = llm.model_cfg.get_num_kv_heads(None)
    head_size = llm.model_cfg.get_head_size()
    num_layers = llm.model_cfg.get_num_layers(None)
    
    kv_caches = [(torch.zeros((128, 16, 1, 512), device="cuda", dtype=torch.float16),
                  torch.zeros((128, 16, 1, 512), device="cuda", dtype=torch.float16)) for _ in range(num_layers)]
    attn_meta = {"slot_mapping": torch.arange(batch_size, device="cuda", dtype=torch.int32), 
                 "seq_lens": torch.ones(batch_size, device="cuda", dtype=torch.int32)}
    
    print("Warmup...")
    for _ in range(5):
        with torch.inference_mode():
            _ = model(input_ids, positions, kv_caches, attn_meta, lora_mapping=lora_mapping)
    torch.cuda.synchronize()

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
    print(f"MOE-LORA BS={batch_size} RESULT")
    print(f"Throughput: {tps:.2f} tokens/sec")
    print(f"Latency:    {latency:.2f} ms")
    print(f"Base MoE:   ~906 TPS")
    print(f"Overhead:   {((906-tps)/906)*100:.1f}%")
    print(f"======================================")
    return tps

if __name__ == "__main__":
    run_moe_lora_benchmark(batch_size=32)
