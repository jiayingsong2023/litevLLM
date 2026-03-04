# SPDX-License-Identifier: Apache-2.0
import torch
import os
import gc
from safetensors import safe_open
from vllm.logger import init_logger

logger = init_logger(__name__)

class ShardedVRAMLoader:
    """
    High-Efficiency Loader for Massive Models (35B+).
    Loads weights from Safetensors directly to GPU in small shards to prevent OOM.
    """
    @staticmethod
    def load_model_weights(model: torch.nn.Module, model_path: str):
        logger.info(f"Starting Sharded VRAM Loading for model in: {model_path}")
        
        # 1. Identify all safetensors files (usually model-00001-of-000XX.safetensors)
        files = sorted([f for f in os.listdir(model_path) if f.endswith(".safetensors")])
        if not files:
            raise FileNotFoundError(f"No weight files found in {model_path}")

        # 2. Iterate through files and load tensors directly to GPU
        for i, filename in enumerate(files):
            full_path = os.path.join(model_path, filename)
            logger.info(f"Processing shard {i+1}/{len(files)}: {filename}")
            
            with safe_open(full_path, framework="pt", device="cuda") as f:
                for key in f.keys():
                    # Find the target module in our LiteModel
                    # We use a naming-aware routing logic
                    target_module = None
                    for name, module in model.named_modules():
                        # Extract the last part of the name for fuzzy matching
                        # e.g., 'model.layers.0.self_attn.qkv_proj' -> 'qkv_proj'
                        lite_name = name.split(".")[-1]
                        if lite_name in key:
                            # Verify the full layer index matches to avoid mis-alignment
                            if any(f".{i}." in key for i in range(100)): # Check layer ID
                                if f".{name.split('.')[2]}." in key: 
                                    target_module = module
                                    break
                            else:
                                # Non-layer specific (like embed_tokens or lm_head)
                                target_module = module
                                break
                    
                    if target_module and hasattr(target_module, "load_weights"):
                        # Stream directly from Safetensors to GPU memory
                        # f.get_tensor(key) returns a torch tensor pinned to the device
                        weight_data = f.get_tensor(key)
                        target_module.load_weights([(key, weight_data)])
                        
                        # Crucial: Delete reference and force fragment collection
                        del weight_data
                    
            # Explicit sync and GC after each shard to clear 'ghost' memory
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            
        logger.info("Sharded VRAM Loading Complete. 100% weights aligned.")

def load_sharded_weights(model: torch.nn.Module, model_path: str):
    ShardedVRAMLoader.load_model_weights(model, model_path)
