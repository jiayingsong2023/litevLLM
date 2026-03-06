# SPDX-License-Identifier: Apache-2.0
import torch
import time
import os
import gc
import json
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.layers.quantization.gguf import _GLOBAL_GGUF_CACHE, clear_gguf_cache

def run_real_awq_inference(model_name, model_path, batch_size=32):
    print(f"\n>>> [REAL AWQ] Starting Inference Verification: {model_name}")
    print(f">>> Path: {model_path}, BS: {batch_size}")
    
    clear_gguf_cache()
    gc.collect()
    torch.cuda.empty_cache()
    
    with open(os.path.join(model_path, "config.json"), "r") as f:
        data = json.load(f)
    
    text_config = data.get("text_config", data)
    if "architectures" not in text_config:
        text_config["architectures"] = data.get("architectures", [])
    
    # SAFETY CACHE LIMIT: 
    # For 35B (20GB weights), 128 blocks (~10GB) is the stable sweet spot on 60GB card.
    if "35B" in model_name:
        _GLOBAL_GGUF_CACHE.max_size = 128
    else:
        _GLOBAL_GGUF_CACHE.max_size = 256

    class LiteVllmConfig:
        def __init__(self, path, hf_cfg):
            # Qwen3.5-35B hidden_size is 2048
            self.hidden_size = hf_cfg.get("hidden_size", 4096)
            n_heads = hf_cfg.get("num_attention_heads", 32)
            self.model_config = type('obj', (object,), {
                'hf_config': type('obj', (object,), hf_cfg),
                'dtype': torch.float16,
                'max_model_len': 2048,
                'model': path,
                'get_num_kv_heads': lambda x: hf_cfg.get("num_key_value_heads", n_heads),
                'get_head_size': lambda: self.hidden_size // n_heads,
                'get_num_layers': lambda x: hf_cfg.get("num_hidden_layers", 32),
            })
            self.parallel_config = type('obj', (object,), {'tensor_parallel_size': 1, 'world_size': 1})
            from vllm.model_executor.layers.quantization.awq import AWQConfig
            # Qwen3.5 uses group_size 32 for AWQ
            g_size = hf_cfg.get("quantization_config", {}).get("group_size", 32)
            self.quant_config = AWQConfig(weight_bits=4, group_size=g_size, zero_point=True)

    v_config = LiteVllmConfig(model_path, text_config)
    hidden_size = v_config.hidden_size
    num_layers = text_config.get("num_hidden_layers", 32)
    
    print(f"Loading model (Hidden={hidden_size}, Layers={num_layers})...")
    start_load = time.time()
    model = get_model(v_config).cuda().half()
    print(f">>> Model loaded in {time.time() - start_load:.2f} seconds.")
    
    input_ids = torch.randint(0, 32000, (batch_size, 1), device="cuda")
    positions = torch.zeros(batch_size, device="cuda", dtype=torch.long)
    
    # PagedAttention Metadata
    attn_metadata = {
        "slot_mapping": torch.arange(batch_size, device="cuda", dtype=torch.int32),
        "seq_lens": torch.ones(batch_size, device="cuda", dtype=torch.int32) * 10,
    }
    
    # Adaptive KV Cache
    kv_caches = []
    # Qwen3.5-35B Linear layers produce different head counts, but Triton PagedAttn
    # needs a stable buffer. We use hidden_size as a safe upper bound.
    for _ in range(num_layers):
        k = torch.zeros((128, 16, 32, 128), device="cuda", dtype=torch.float16) # Generic buffer
        v = torch.zeros((128, 16, 32, 128), device="cuda", dtype=torch.float16)
        kv_caches.append((k, v))

    print("Warmup...")
    try:
        for _ in range(2):
            with torch.inference_mode():
                _ = model(input_ids, positions, kv_caches, attn_metadata)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"Warmup failed: {e}")
        return 0

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
    print(f"REAL AWQ PERFORMANCE: {model_name}")
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
    # Test 9B
    path_9b = "models/Qwen3.5-9B-AWQ"
    if os.path.exists(path_9b) and any(f.endswith(".safetensors") for f in os.listdir(path_9b)):
        run_real_awq_inference("Qwen3.5-9B", path_9b, batch_size=32)
    
    # Test 35B
    path_35b = "models/Qwen3.5-35B-AWQ"
    if os.path.exists(path_35b) and any(f.endswith(".safetensors") for f in os.listdir(path_35b)):
        # Run 35B with BS=1 for absolute confirmation
        run_real_awq_inference("Qwen3.5-35B-BS1", path_35b, batch_size=1)
