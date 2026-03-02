# SPDX-License-Identifier: Apache-2.0
import torch
import time
from vllm.model_executor.models.llama import LlamaModel

def benchmark_multi_lora_concurrency():
    print("--- LitevLLM Multi-adapter Concurrency Benchmark ---")
    
    # 1. Setup Config (Small 2-layer Llama for speed)
    class DummyHFConfig:
        def __init__(self):
            self.num_hidden_layers = 2
            self.hidden_size = 2048
            self.num_attention_heads = 32
            self.num_key_value_heads = 8
            self.max_position_embeddings = 2048
            self.rms_norm_eps = 1e-6
            self.vocab_size = 32000
            self.intermediate_size = 5504
            self.architectures = ["LlamaForCausalLM"]

    class DummyVllmConfig:
        def __init__(self):
            self.model_config = type('obj', (object,), {
                'hf_config': DummyHFConfig(),
                'dtype': torch.float16,
                'max_model_len': 2048,
                'model': "dummy"
            })
            self.parallel_config = type('obj', (object,), {})
            self.quant_config = None

    config = DummyVllmConfig()
    model = LlamaModel(config).cuda().half()
    
    # 2. 模拟 3 个不同的 LoRA 适配器
    rank = 16
    adapter_ids = [101, 102, 103]
    print(f"Injecting {len(adapter_ids)} unique adapters into LiteLinear layers...")
    
    for module in model.modules():
        from vllm.model_executor.layers.lite_linear import LiteLinear
        if isinstance(module, LiteLinear):
            for aid in adapter_ids:
                la = torch.randn(rank, module.input_size, device="cuda", dtype=torch.float16)
                lb = torch.randn(module.output_size, rank, device="cuda", dtype=torch.float16)
                module.add_adapter(aid, la, lb)

    # 3. 准备混合 Batch (Batch Size = 32)
    batch_size = 32
    input_ids = torch.ones((batch_size, 1), device="cuda", dtype=torch.long)
    positions = torch.zeros(batch_size, device="cuda", dtype=torch.long) + 10
    
    # 模拟分配：10 tokens 用 A, 10 tokens 用 B, 10 tokens 用 C, 2 tokens 用 Base
    lora_mapping = torch.zeros(batch_size, device="cuda", dtype=torch.int32)
    lora_mapping[0:10] = 101
    lora_mapping[10:20] = 102
    lora_mapping[20:30] = 103
    
    kv_caches = [(torch.zeros(128, 16, 8, 64, device="cuda", dtype=torch.float16), 
                  torch.zeros(128, 16, 8, 64, device="cuda", dtype=torch.float16)) for _ in range(2)]
    
    attn_metadata = {
        "slot_mapping": torch.arange(batch_size, device="cuda", dtype=torch.int32),
        "seq_lens": torch.ones(batch_size, device="cuda", dtype=torch.int32) * 11,
        "block_tables": torch.zeros((batch_size, 128), device="cuda", dtype=torch.int32)
    }

    # 4. Benchmark
    iters = 20
    print(f"Running Multi-adapter benchmark (BS={batch_size})...")
    start_time = time.time()
    for _ in range(iters):
        with torch.inference_mode():
            model(input_ids, positions, kv_caches, attn_metadata, lora_mapping=lora_mapping)
    torch.cuda.synchronize()
    
    avg_latency = (time.time() - start_time) / iters * 1000
    print(f"\nRESULT: Avg Latency per step = {avg_latency:.2f} ms")
    print(f"Total Concurrent Throughput = {(1000 / avg_latency) * batch_size:.2f} tokens/sec")
    print("-------------------------------------------------------")

if __name__ == "__main__":
    if torch.cuda.is_available():
        benchmark_multi_lora_concurrency()
