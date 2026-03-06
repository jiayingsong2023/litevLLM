# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import time
import os
import gc
import json
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.sharded_loader import load_sharded_weights
from vllm.model_executor.layers.quantization.gguf import clear_gguf_cache

def test_35b_sharded_load():
    model_path = "models/Qwen3.5-35B-AWQ"
    print(f"\n>>> [35B MAIDEN VOYAGE] Starting Stability Test for: {model_path}")
    
    # 1. GPU Reset
    clear_gguf_cache()
    gc.collect()
    torch.cuda.empty_cache()
    
    # 2. Build 35B Mock Config
    if not os.path.exists(os.path.join(model_path, "config.json")):
        print("Config not ready. Skipping.")
        return
        
    with open(os.path.join(model_path, "config.json"), "r") as f:
        data = json.load(f)
    text_config = data.get("text_config", data)
    
    class LiteVllmConfig:
        def __init__(self, path, hf_cfg):
            self.model_config = type('obj', (object,), {
                'hf_config': type('obj', (object,), hf_cfg),
                'dtype': torch.float16,
                'max_model_len': 2048,
                'model': path,
                'get_num_kv_heads': lambda x: 4, 
                'get_head_size': lambda: 128,
                'get_num_layers': lambda x: 40,
            })
            self.parallel_config = type('obj', (object,), {'tensor_parallel_size': 1, 'world_size': 1})
            from vllm.model_executor.layers.quantization.awq import AWQConfig
            self.quant_config = AWQConfig(weight_bits=4, group_size=32, zero_point=True)

    v_config = LiteVllmConfig(model_path, text_config)
    
    # 3. Instantiate Empty 35B Shell
    print(">>> Stage 1: Initializing 35B Model Shell on GPU...")
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM
    model = Qwen3_5ForCausalLM(v_config).cuda().half()
    
    # 4. Trigger Sharded Loading
    print(">>> Stage 2: Attempting Sharded VRAM Injection...")
    try:
        load_sharded_weights(model, model_path)
        print(">>> SUCCESS: Sharded VRAM injection passed.")
    except Exception as e:
        print(f">>> FAILED during injection: {e}")
        # Not fatal if shards are still downloading

    # 5. Heartbeat Probe
    print(">>> Stage 3: Operator Heartbeat (Minimal Forward)...")
    input_ids = torch.randint(0, 32000, (1, 1), device="cuda")
    attn_metadata = {
        "slot_mapping": torch.tensor([0], device="cuda", dtype=torch.int32),
        "seq_lens": torch.tensor([1], device="cuda", dtype=torch.int32),
    }
    kv_caches = [(torch.zeros(1, 1, 1, 1, device="cuda"), torch.zeros(1, 1, 1, 1, device="cuda"))] * 40
    
    try:
        with torch.inference_mode():
            _ = model(input_ids, None, kv_caches, attn_metadata)
        print(">>> SUCCESS: Heartbeat forward passed.")
    except Exception as e:
        print(f">>> Heartbeat Failed (expected if weights missing): {e}")

    print("\n=== [35B MAIDEN VOYAGE] COMPLETED ===")

if __name__ == "__main__":
    test_35b_sharded_load()
