# SPDX-License-Identifier: Apache-2.0
import torch
import time
from vllm.model_executor.model_loader import get_model

def run_e2e_lora_perf():
    print("--- E2E LoRA Performance Benchmark: TinyLlama with Rank 16 Adapter ---")
    model_id = "models/TinyLlama-1.1B-Chat-v1.0"
    rank = 16
    
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
    
    # 2. Load Model
    model = get_model(v_config).cuda().half()
    
    # 3. Inject LoRA Weights into all compatible layers
    print(f"Injecting Rank {rank} LoRA adapters into all layers...")
    for name, module in model.named_modules():
        from vllm.model_executor.layers.lite_linear import LiteLinear
        if isinstance(module, LiteLinear):
            lora_a = torch.randn(rank, module.input_size, device="cuda", dtype=torch.float16)
            lora_b = torch.randn(module.output_size, rank, device="cuda", dtype=torch.float16)
            module.set_lora(lora_a, lora_b, scaling=2.0)

    # 4. Prepare Inputs (Batch=1, Decode)
    input_ids = torch.tensor([[1]], device="cuda")
    positions = torch.tensor([10], device="cuda")
    
    num_heads = hf_config.num_attention_heads
    num_kv_heads = hf_config.num_key_value_heads
    head_size = hf_config.hidden_size // num_heads
    
    kv_caches = []
    for _ in range(hf_config.num_hidden_layers):
        k_cache = torch.zeros((16, 16, num_kv_heads, head_size), device="cuda", dtype=torch.float16)
        v_cache = torch.zeros((16, 16, num_kv_heads, head_size), device="cuda", dtype=torch.float16)
        kv_caches.append((k_cache, v_cache))

    attn_metadata = {
        "slot_mapping": torch.tensor([10], device="cuda", dtype=torch.int32),
        "seq_lens": torch.tensor([11], device="cuda", dtype=torch.int32),
        "block_tables": torch.zeros((1, 16), device="cuda", dtype=torch.int32)
    }

    # 5. Warmup
    print("Warmup...")
    for _ in range(10):
        with torch.inference_mode():
            model(input_ids, positions, kv_caches, attn_metadata)
    torch.cuda.synchronize()

    # 6. Benchmark
    iters = 100
    print(f"Running {iters} iterations...")
    start_time = time.time()
    for _ in range(iters):
        with torch.inference_mode():
            model(input_ids, positions, kv_caches, attn_metadata)
    torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    avg_latency = (total_time / iters) * 1000
    tps = 1000 / avg_latency
    
    print("\n--- LoRA E2E Benchmark Result ---")
    print(f"Model:        {model_id}")
    print(f"Adapter:      Rank {rank}")
    print(f"Avg Latency:  {avg_latency:.2f} ms per token")
    print(f"Throughput:   {tps:.2f} tokens/sec")
    print("---------------------------------")

if __name__ == "__main__":
    run_e2e_lora_perf()
