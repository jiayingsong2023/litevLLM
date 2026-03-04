# SPDX-License-Identifier: Apache-2.0
import torch
import time
import os
import json
import gc
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.layers.quantization.gguf import clear_gguf_cache

def benchmark_real_multimodal(model_path, batch_size=32, image_tokens=576):
    print(f"\n>>> [REAL] Benchmarking Multi-modal: {model_path}")
    print(f">>> Image Tokens: {image_tokens}, Batch Size: {batch_size}")
    
    clear_gguf_cache()
    gc.collect()
    
    # 1. Load Real Config
    config_file = os.path.join(model_path, "config.json")
    with open(config_file, "r") as f:
        config_data = json.load(f)
    text_config = config_data.get("text_config", config_data)
    
    class FakeVllmConfig:
        def __init__(self):
            self.model_config = type('obj', (object,), {
                'hf_config': type('obj', (object,), {**text_config, 'architectures': config_data.get("architectures", [])}),
                'dtype': torch.float16,
                'max_model_len': 4096,
                'model': model_path,
                'get_num_kv_heads': lambda x: text_config.get("num_key_value_heads", text_config["num_attention_heads"]),
                'get_head_size': lambda: text_config["hidden_size"] // text_config["num_attention_heads"],
                'get_num_layers': lambda x: text_config["num_hidden_layers"]
            })
            self.parallel_config = type('obj', (object,), {'tensor_parallel_size': 1, 'world_size': 1})
            from vllm.model_executor.layers.quantization.gguf import GGUFConfig
            self.quant_config = GGUFConfig()

    v_config = FakeVllmConfig()
    model = get_model(v_config).cuda().half()
    
    # 2. Prepare Multi-modal Inputs
    # Simulate a mix of text and vision tokens
    # input_ids shape: [batch, total_seq]
    total_seq = 1 + image_tokens # 1 text prompt + image grid
    input_ids = torch.ones((batch_size, total_seq), device="cuda", dtype=torch.long)
    positions = torch.arange(total_seq, device="cuda").repeat(batch_size, 1)
    
    # Standard PagedAttention setup
    num_kv_heads = text_config.get("num_key_value_heads", text_config["num_attention_heads"])
    head_size = text_config["hidden_size"] // text_config["num_attention_heads"]
    kv_caches = []
    for _ in range(text_config["num_hidden_layers"]):
        k = torch.zeros((512, 16, num_kv_heads, head_size), device="cuda", dtype=torch.float16)
        v = torch.zeros((512, 16, num_kv_heads, head_size), device="cuda", dtype=torch.float16)
        kv_caches.append((k, v))

    attn_metadata = {
        "slot_mapping": torch.arange(batch_size * total_seq, device="cuda", dtype=torch.int32),
        "seq_lens": torch.ones(batch_size, device="cuda", dtype=torch.int32) * total_seq,
        "block_tables": torch.zeros((batch_size, 512), device="cuda", dtype=torch.int32)
    }

    # 3. Warmup
    print("Warmup...")
    for _ in range(3):
        with torch.inference_mode():
            model(input_ids, positions, kv_caches, attn_metadata)
    
    # 4. Benchmark
    iters = 10
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        with torch.inference_mode():
            model(input_ids, positions, kv_caches, attn_metadata)
    torch.cuda.synchronize()
    
    tps = (batch_size * total_seq * iters) / (time.time() - start)
    print(f"RESULT: {tps:.2f} total tokens/sec (Prefill-style)")
    
    del model, kv_caches
    clear_gguf_cache()
    gc.collect()
    return tps

if __name__ == "__main__":
    path = "models/Qwen3.5-9B-GGUF"
    if os.path.exists(path):
        benchmark_real_multimodal(path)
    else:
        print(f"Path not found: {path}")
