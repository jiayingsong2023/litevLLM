# SPDX-License-Identifier: Apache-2.0
import torch
import time
from vllm.model_executor.models.qwen2_vl import Qwen2VLForCausalLM

def benchmark_multimodal_scaling():
    print("--- LitevLLM Multi-Modal Performance Benchmark ---")
    
    # 1. Setup Config
    class DummyHFConfig:
        def __init__(self):
            self.num_hidden_layers = 24
            self.hidden_size = 2048
            self.num_attention_heads = 16
            self.num_key_value_heads = 16
            self.max_position_embeddings = 4096
            self.vocab_size = 32000
            self.rms_norm_eps = 1e-6
            self.intermediate_size = 5504
            self.model = "qwen2-vl-7b-sim"
    
    class FakeVllmConfig:
        def __init__(self):
            self.model_config = type('obj', (object,), {
                'hf_config': DummyHFConfig(),
                'dtype': torch.float16,
                'max_model_len': 4096,
                'model': "qwen2-vl-sim"
            })
            self.parallel_config = type('obj', (object,), {})
            self.quant_config = None

    v_config = FakeVllmConfig()
    model = Qwen2VLForCausalLM(v_config).cuda().half()
    
    # 2. 模拟高负载多模态输入 (Batch=32)
    batch_size = 32
    num_vision_tokens = 576 # 典型的 Qwen2-VL 224x224 图像 token 数
    
    # 模拟 Prefill 场景：每个请求包含 576 个视觉 token
    input_ids = torch.ones((batch_size, 1), device="cuda", dtype=torch.long)
    positions = torch.zeros(batch_size, device="cuda", dtype=torch.long) + num_vision_tokens
    
    # KV Cache 模拟
    kv_caches = [(torch.zeros(2048, 16, 16, 128, device="cuda", dtype=torch.float16), 
                  torch.zeros(2048, 16, 16, 128, device="cuda", dtype=torch.float16)) for _ in range(24)]
    
    attn_metadata = {
        "slot_mapping": torch.arange(batch_size, device="cuda", dtype=torch.int32),
        "seq_lens": torch.ones(batch_size, device="cuda", dtype=torch.int32) * (num_vision_tokens + 1),
        "block_tables": torch.zeros((batch_size, 128), device="cuda", dtype=torch.int32)
    }

    # 3. Warmup
    print(f"Warming up with Batch Size {batch_size} and {num_vision_tokens} vision tokens...")
    for _ in range(5):
        with torch.inference_mode():
            model(input_ids, positions, kv_caches, attn_metadata)
    torch.cuda.synchronize()

    # 4. Benchmark Decode TPS
    iters = 20
    print(f"Benchmarking {iters} iterations...")
    start_time = time.time()
    for _ in range(iters):
        with torch.inference_mode():
            model(input_ids, positions, kv_caches, attn_metadata)
    torch.cuda.synchronize()
    end_time = time.time()

    avg_latency = (end_time - start_time) / iters * 1000
    total_tps = (1000 / avg_latency) * batch_size
    
    print(f"\n--- Multimodal Results (Batch {batch_size}, 576 Vision Context) ---")
    print(f"Avg Latency per step: {avg_latency:.2f} ms")
    print(f"Total Throughput:     {total_tps:.2f} tokens/sec")
    print("-----------------------------------------------------------------")

if __name__ == "__main__":
    if torch.cuda.is_available():
        benchmark_multimodal_scaling()
