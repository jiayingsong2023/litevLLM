# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import os
import gc
import re
from safetensors import safe_open
from vllm.logger import init_logger

logger = init_logger(__name__)

class ShardedVRAMLoader:
    """
    Production-grade Sharded VRAM Loader.
    Handlesprefix remapping and MoE expert slicing for 35B+ models.
    """
    MODULE_MAP = {
        "language_model.layers": "layers",
        "linear_attn": "self_attn",
        "in_proj_qkv": "qkv_proj",
        "in_proj_a": "q_proj",
        "in_proj_b": "kv_proj",
        "out_proj": "o_proj"
    }

    @staticmethod
    def load_model_weights(model: torch.nn.Module, model_path: str):
        logger.info(f">>> Sharded loading from: {model_path}")
        
        files = sorted([f for f in os.listdir(model_path) if f.endswith(".safetensors")])
        if not files:
            logger.warning(f"No safetensors found in {model_path}. Weights might be missing.")
            return

        # Map LiteModel modules for fast lookup
        lite_modules = {}
        for name, module in model.named_modules():
            if hasattr(module, "load_weights"):
                clean_name = name[6:] if name.startswith("model.") else name
                lite_modules[clean_name] = module

        for filename in files:
            full_path = os.path.join(model_path, filename)
            logger.info(f"Loading shard: {filename}")
            
            try:
                with safe_open(full_path, framework="pt", device="cuda") as f:
                    for hf_key in f.keys():
                        tensor = f.get_tensor(hf_key)
                        
                        # Remap Key
                        lite_key = hf_key
                        if lite_key.startswith("model."): lite_key = lite_key[6:]
                        for old, new in ShardedVRAMLoader.MODULE_MAP.items():
                            lite_key = lite_key.replace(old, new)
                        
                        found = False
                        # 1. MoE Experts
                        expert_match = re.search(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(.*)", lite_key)
                        if expert_match:
                            l_idx, e_idx = int(expert_match.group(1)), int(expert_match.group(2))
                            target = f"layers.{l_idx}.mlp.experts"
                            if target in lite_modules:
                                lite_modules[target].load_weights([(hf_key, tensor)], expert_idx=e_idx)
                                found = True

                        # 2. Standard Modules
                        if not found:
                            for mod_name, module in lite_modules.items():
                                if lite_key == mod_name or lite_key.startswith(mod_name + "."):
                                    module.load_weights([(hf_key, tensor)])
                                    found = True
                                    break
                        
                        # 3. Direct Params (Norms/Embed)
                        if not found:
                            for param_name, param in model.named_parameters():
                                clean_p = param_name[6:] if param_name.startswith("model.") else param_name
                                if clean_p == lite_key:
                                    param.data.copy_(tensor)
                                    found = True
                                    break
                        del tensor
            except Exception as e:
                logger.error(f"Error loading shard {filename}: {e}")
                continue

            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            
        logger.info(">>> Sharded Loading Sequence Complete.")

def load_sharded_weights(model: torch.nn.Module, model_path: str):
    ShardedVRAMLoader.load_model_weights(model, model_path)
