# SPDX-License-Identifier: Apache-2.0
import torch
import time
import os
from vllm.model_executor.model_loader import get_model
from transformers import AutoConfig

def run_e2e_perf():
    model_id = "models/TinyLlama-1.1B-Chat-v1.0"
    if not os.path.exists(model_id):
        print("Model not found.")
        return

    print(f"--- E2E Performance Benchmark: {model_id} ---")
    
    # 1. Load Model
    hf_config = AutoConfig.from_pretrained(model_id)
    class FakeVllmConfig:
        def __init__(self):
            self.model_config = type('obj', (object,), {
                'hf_config': hf_config,
                'dtype': torch.float16,
                'max_model_len': 2048
            })
            self.quant_config = None
    
    model = get_model(FakeVllmConfig()).cuda().half()
    
    # 2. Prepare Inputs (Batch=1, Decode phase)
    batch_size = 1
    input_ids = torch.tensor([[1]], device="cuda")
    positions = torch.tensor([10], device="cuda")
    kv_caches = [torch.zeros(1, 8, 128, 128, device="cuda") for _ in range(hf_config.num_hidden_layers)]
    attn_metadata = {
        "slot_mapping": torch.tensor([10], device="cuda"),
        "seq_lens": [11]
    }

    # 3. Warmup
    print("Warmup...")
    for _ in range(10):
        model(input_ids, positions, kv_caches, attn_metadata)
    torch.cuda.synchronize()

    # 4. Benchmark
    iters = 100
    print(f"Running {iters} iterations...")
    start_time = time.time()
    for _ in range(iters):
        model(input_ids, positions, kv_caches, attn_metadata)
    torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    avg_latency = (total_time / iters) * 1000 # ms
    tps = 1000 / avg_latency
    
    print("\n--- Benchmark Result ---")
    print(f"Avg Latency: {avg_latency:.2f} ms per token")
    print(f"Throughput:  {tps:.2f} tokens/sec")
    print("------------------------")

if __name__ == "__main__":
    run_e2e_perf()
