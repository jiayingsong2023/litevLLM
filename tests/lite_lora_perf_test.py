# SPDX-License-Identifier: Apache-2.0
import torch
import time
from vllm.model_executor.models.llama import LlamaModel

def benchmark_lora_overhead():
    print("--- LitevLLM LoRA Overhead Benchmark ---")
    
    # 1. Setup Config
    class DummyHFConfig:
        def __init__(self):
            self.num_hidden_layers = 2
            self.hidden_size = 2048
            self.num_attention_heads = 32
            self.num_key_value_heads = 8
            self.max_position_embeddings = 2048
            self.rms_norm_eps = 1e-6
            self.vocab_size = 32000
            self.intermediate_size = 5504 # Standard Llama-like ratio
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
    
    # Prepare Inputs
    input_ids = torch.tensor([[1]], device="cuda")
    positions = torch.tensor([0], device="cuda")
    kv_caches = [(torch.zeros(16, 16, 8, 64, device="cuda", dtype=torch.float16), 
                  torch.zeros(16, 16, 8, 64, device="cuda", dtype=torch.float16)) for _ in range(2)]
    attn_metadata = {
        "slot_mapping": torch.tensor([0], device="cuda", dtype=torch.int32),
        "seq_lens": torch.tensor([1], device="cuda", dtype=torch.int32),
        "block_tables": torch.zeros((1, 16), device="cuda", dtype=torch.int32)
    }

    # --- Scenario 1: Base Model (No LoRA) ---
    print("\n[Scenario 1] Base Model (No LoRA)")
    for _ in range(5): model(input_ids, positions, kv_caches, attn_metadata)
    torch.cuda.synchronize()
    
    iters = 100
    start = time.time()
    for _ in range(iters): model(input_ids, positions, kv_caches, attn_metadata)
    torch.cuda.synchronize()
    base_time = (time.time() - start) / iters * 1000
    print(f"Base Latency: {base_time:.4f} ms")

    # --- Scenario 2: LoRA Enabled (Rank 16) ---
    print("\n[Scenario 2] LoRA Enabled (Rank 16)")
    rank = 16
    for layer in model.layers:
        # qkv_proj output size is (num_heads + 2*num_kv_heads) * head_dim
        # 32*64 + 2*8*64 = 2048 + 1024 = 3072
        layer.self_attn.qkv_proj.set_lora(
            torch.randn(rank, 2048, device="cuda", dtype=torch.float16),
            torch.randn(3072, rank, device="cuda", dtype=torch.float16)
        )
        # gate_up_proj output size is 2 * intermediate_size
        # 2 * 5504 = 11008
        layer.mlp.gate_up_proj.set_lora(
            torch.randn(rank, 2048, device="cuda", dtype=torch.float16),
            torch.randn(11008, rank, device="cuda", dtype=torch.float16)
        )

    for _ in range(5): model(input_ids, positions, kv_caches, attn_metadata)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(iters): model(input_ids, positions, kv_caches, attn_metadata)
    torch.cuda.synchronize()
    lora_time = (time.time() - start) / iters * 1000
    
    overhead = (lora_time - base_time) / base_time * 100
    print(f"LoRA Latency: {lora_time:.4f} ms")
    print(f"Total Overhead: {overhead:.2f}%")
    print("-----------------------------------------")

if __name__ == "__main__":
    benchmark_lora_overhead()
