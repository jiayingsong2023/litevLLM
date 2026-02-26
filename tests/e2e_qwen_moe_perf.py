# SPDX-License-Identifier: Apache-2.0
import torch
import time
import os
from vllm.model_executor.model_loader import get_model
from transformers import AutoConfig

def run_qwen_moe_perf():
    model_id = "models/Qwen1.5-MoE-A2.7B-Chat"
    if not os.path.exists(model_id):
        print(f"Model path {model_id} not found.")
        return

    print(f"--- E2E Performance Benchmark: {model_id} ---")
    
    # 1. Load Model Config
    hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    
    class FakeVllmConfig:
        def __init__(self):
            self.model_config = type('obj', (object,), {
                'hf_config': hf_config,
                'dtype': torch.float16,
                'max_model_len': 2048
            })
            self.quant_config = None
    
    # 2. Instantiate Model
    print("Loading model and weights...")
    model = get_model(FakeVllmConfig()).cuda().half()
    
    # 3. Prepare Inputs (Batch=1, Decode phase)
    batch_size = 1
    input_ids = torch.tensor([[1]], device="cuda")
    positions = torch.tensor([10], device="cuda")
    # Mock empty KV cache
    kv_caches = [torch.zeros(1, hf_config.num_key_value_heads, 128, 128, device="cuda") 
                 for _ in range(hf_config.num_hidden_layers)]
    attn_metadata = {
        "slot_mapping": torch.tensor([10], device="cuda"),
        "seq_lens": [11]
    }

    # 4. Warmup
    print("Warmup...")
    for _ in range(5):
        with torch.inference_mode():
            model(input_ids, positions, kv_caches, attn_metadata)
    torch.cuda.synchronize()

    # 5. Benchmark
    iters = 50
    print(f"Running {iters} iterations...")
    start_time = time.time()
    for _ in range(iters):
        with torch.inference_mode():
            model(input_ids, positions, kv_caches, attn_metadata)
    torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    avg_latency = (total_time / iters) * 1000 # ms
    tps = 1000 / avg_latency
    
    print("\n--- Benchmark Result ---")
    print(f"Model:        {model_id}")
    print(f"Avg Latency:  {avg_latency:.2f} ms per token")
    print(f"Throughput:   {tps:.2f} tokens/sec")
    print("------------------------")

if __name__ == "__main__":
    run_qwen_moe_perf()
