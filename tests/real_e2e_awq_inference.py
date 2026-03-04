# SPDX-License-Identifier: Apache-2.0
import torch
import time
import os
import gc
import json
from vllm.model_executor.model_loader import get_model, get_tokenizer
from vllm.model_executor.layers.quantization.gguf import clear_gguf_cache

def run_real_awq_inference(model_path, batch_size=32):
    print(f"\n>>> [REAL AWQ] Starting Inference Verification")
    print(f">>> Path: {model_path}, BS: {batch_size}")
    
    # 1. Reset GPU state
    clear_gguf_cache()
    gc.collect()
    torch.cuda.empty_cache()
    
    # 2. Load config.json manually to get full metadata
    with open(os.path.join(model_path, "config.json"), "r") as f:
        data = json.load(f)
    
    text_config = data.get("text_config", data)
    # Ensure architectures is present in the config passed to get_model
    if "architectures" not in text_config:
        text_config["architectures"] = data.get("architectures", [])
    
    num_layers = text_config.get("num_hidden_layers", 32)
    num_heads = text_config.get("num_attention_heads", 32)
    num_kv_heads = text_config.get("num_key_value_heads", num_heads)
    hidden_size = text_config.get("hidden_size", 4096)
    head_size = hidden_size // num_heads

    class LiteVllmConfig:
        def __init__(self, path, hf_cfg):
            self.model_config = type('obj', (object,), {
                'hf_config': type('obj', (object,), hf_cfg),
                'dtype': torch.float16,
                'max_model_len': 4096,
                'model': path,
                'get_num_kv_heads': lambda x: num_kv_heads,
                'get_head_size': lambda: head_size,
                'get_num_layers': lambda x: num_layers,
            })
            self.parallel_config = type('obj', (object,), {'tensor_parallel_size': 1, 'world_size': 1})
            from vllm.model_executor.layers.quantization.awq import AWQConfig
            # Default AWQ config for Qwen
            self.quant_config = AWQConfig(weight_bits=4, group_size=32, zero_point=True)

    v_config = LiteVllmConfig(model_path, text_config)
    
    # 3. Load Model (This triggers Safetensors Aligner)
    print("Loading model and aligning Safetensors weights...")
    start_load = time.time()
    model = get_model(v_config).cuda().half()
    print(f">>> Model loaded in {time.time() - start_load:.2f} seconds.")
    
    # 4. Prepare Inputs
    input_ids = torch.randint(0, 32000, (batch_size, 1), device="cuda")
    positions = torch.zeros(batch_size, device="cuda", dtype=torch.long)
    
    # Standard PagedAttention metadata (Mocked for speed test)
    attn_metadata = {
        "slot_mapping": torch.arange(batch_size, device="cuda", dtype=torch.int32),
        "seq_lens": torch.ones(batch_size, device="cuda", dtype=torch.int32) * 10,
    }
    
    # KV cache setup
    kv_caches = []
    num_blocks = 128
    for _ in range(num_layers):
        k = torch.zeros((num_blocks, 16, num_kv_heads, head_size), device="cuda", dtype=torch.float16)
        v = torch.zeros((num_blocks, 16, num_kv_heads, head_size), device="cuda", dtype=torch.float16)
        kv_caches.append((k, v))

    # 5. Warmup
    print("Warmup...")
    for _ in range(3):
        with torch.inference_mode():
            _ = model(input_ids, positions, kv_caches, attn_metadata)
    torch.cuda.synchronize()
    
    # 6. Benchmark
    iters = 10
    print(f"Running {iters} iterations...")
    start_time = time.time()
    for _ in range(iters):
        with torch.inference_mode():
            _ = model(input_ids, positions, kv_caches, attn_metadata)
    torch.cuda.synchronize()
    end_time = time.time()
    
    latency = (end_time - start_time) / iters * 1000
    tps = (1000 / latency) * batch_size
    
    print(f"\n======================================")
    print(f"REAL AWQ PERFORMANCE RESULT")
    print(f"======================================")
    print(f"Batch Size: {batch_size}")
    print(f"Avg Latency: {latency:.2f} ms")
    print(f"Throughput:  {tps:.2f} tokens/sec")
    print(f"======================================")
    
    del model, kv_caches
    clear_gguf_cache()
    gc.collect()
    torch.cuda.empty_cache()
    return tps

if __name__ == "__main__":
    path = "models/Qwen3.5-9B-AWQ"
    # Check if any safetensors files exist
    if os.path.exists(path) and any(f.endswith(".safetensors") for f in os.listdir(path)):
        run_real_awq_inference(path, batch_size=32)
    else:
        print(f"Model not ready yet: {path}. Please wait for download to finish.")
