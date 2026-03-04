# SPDX-License-Identifier: Apache-2.0
import torch
import time
import os
import gc
from vllm.model_executor.model_loader import get_model
from transformers import AutoConfig

def benchmark_real_lora_scaling(model_path, max_batch_size=128):
    print(f"\n>>> [REAL] Benchmarking LoRA Batch Scaling: {model_path}")
    
    # 1. Load Real Config
    hf_config = AutoConfig.from_pretrained(model_path)
    
    class FakeVllmConfig:
        def __init__(self):
            self.model_config = type('obj', (object,), {
                'hf_config': hf_config,
                'dtype': torch.float16,
                'max_model_len': 2048,
                'model': model_path,
                'get_num_kv_heads': lambda x: getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads),
                'get_head_size': lambda: hf_config.hidden_size // hf_config.num_attention_heads,
                'get_num_layers': lambda x: hf_config.num_hidden_layers
            })
            self.parallel_config = type('obj', (object,), {'tensor_parallel_size': 1, 'world_size': 1})
            self.quant_config = None

    v_config = FakeVllmConfig()
    model = get_model(v_config).cuda().half()
    
    # 2. Inject Real LoRA Tensors
    # We create real-shape tensors for the first layer's self_attn to test throughput
    lora_rank = 16
    for name, module in model.named_modules():
        if "qkv_proj" in name or "gate_up_proj" in name:
            in_dim = module.input_size
            out_dim = module.output_size
            # Create real weights on GPU
            la = torch.randn((lora_rank, in_dim), device="cuda", dtype=torch.float16)
            lb = torch.randn((out_dim, lora_rank), device="cuda", dtype=torch.float16)
            module.add_adapter(lora_id=1, lora_a=la, lora_b=lb, scaling=1.0)

    # 3. Scaling Test
    results = {}
    for bs in [1, 8, 32, 64, 128]:
        input_ids = torch.ones((bs, 1), device="cuda", dtype=torch.long)
        positions = torch.zeros(bs, device="cuda", dtype=torch.long) + 10
        # All tokens use lora_id=1
        lora_mapping = torch.ones(bs, device="cuda", dtype=torch.int32)
        
        num_kv_heads = getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads)
        head_size = hf_config.hidden_size // hf_config.num_attention_heads
        
        kv_caches = []
        for _ in range(hf_config.num_hidden_layers):
            k = torch.zeros((512, 16, num_kv_heads, head_size), device="cuda", dtype=torch.float16)
            v = torch.zeros((512, 16, num_kv_heads, head_size), device="cuda", dtype=torch.float16)
            kv_caches.append((k, v))

        attn_metadata = {
            "slot_mapping": torch.arange(bs, device="cuda", dtype=torch.int32),
            "seq_lens": torch.ones(bs, device="cuda", dtype=torch.int32) * 11,
            "block_tables": torch.zeros((bs, 512), device="cuda", dtype=torch.int32)
        }

        # Warmup
        for _ in range(5):
            with torch.inference_mode():
                model(input_ids, positions, kv_caches, attn_metadata, lora_mapping=lora_mapping)
        
        # Benchmark
        iters = 20
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            with torch.inference_mode():
                model(input_ids, positions, kv_caches, attn_metadata, lora_mapping=lora_mapping)
        torch.cuda.synchronize()
        tps = (bs * iters) / (time.time() - start)
        results[bs] = tps
        print(f"BS={bs:3}: {tps:.2f} tokens/sec")

    return results

if __name__ == "__main__":
    path = "models/TinyLlama-1.1B-Chat-v1.0"
    if os.path.exists(path):
        res = benchmark_real_lora_scaling(path)
        print("\n" + "="*40)
        print("REAL LORA SCALING SUMMARY")
        print("="*40)
        for bs, tps in res.items():
            print(f"Batch Size {bs:3}: {tps:8.2f} TPS")
        print("="*40)
    else:
        print(f"Path not found: {path}")
