# SPDX-License-Identifier: Apache-2.0
import torch
import os
import gc
from safetensors import safe_open
from vllm.logger import init_logger

logger = init_logger(__name__)

class SafetensorsAligner:
    """
    Improved Automatic Weight Aligner for Safetensors.
    Handles prefixes, nested structures, and diverse naming conventions.
    """
    @staticmethod
    def load_weights(model: torch.nn.Module, model_path: str):
        logger.info(f">>> Aligning weights from: {model_path}")
        
        files = sorted([f for f in os.listdir(model_path) if f.endswith(".safetensors")])
        if not files:
            raise FileNotFoundError(f"No safetensors found in {model_path}")

        # Build a flat map of model modules for quick lookup
        # We use the full name as the key
        module_map = {}
        for name, module in model.named_modules():
            if hasattr(module, "load_weights"):
                module_map[name] = module

        for filename in files:
            full_path = os.path.join(model_path, filename)
            logger.info(f"Loading shard: {filename}")
            with safe_open(full_path, framework="pt", device="cuda") as f:
                for hf_key in f.keys():
                    tensor = f.get_tensor(hf_key)
                    
                    # 1. Direct match
                    if hf_key in module_map:
                        module_map[hf_key].load_weights([(hf_key, tensor)])
                        continue
                    
                    # 2. Heuristic match: skip common prefixes like 'model.language_model'
                    # and try to match the rest.
                    clean_key = hf_key
                    prefixes_to_skip = ["model.language_model.", "model.", "transformer."]
                    for p in prefixes_to_skip:
                        if clean_key.startswith(p):
                            clean_key = clean_key[len(p):]
                            break
                    
                    # Try to find which module the clean_key belongs to
                    found = False
                    # Optimization: only check modules that could possibly match
                    for mod_name, module in module_map.items():
                        if mod_name in clean_key:
                            # Verify it's a sub-parameter of this module
                            # e.g., 'layers.0.mlp.down_proj' in 'layers.0.mlp.down_proj.weight_packed'
                            module.load_weights([(hf_key, tensor)])
                            found = True
                            break
                    
                    if not found:
                        # Fallback to direct parameter matching for Norms/Embeddings
                        for param_name, param in model.named_parameters():
                            if param_name in hf_key or param_name in clean_key:
                                param.data.copy_(tensor)
                                found = True
                                break
                    
                    del tensor
            
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
        
        logger.info(">>> Weight Alignment Complete.")

def load_safetensors_weights(model: torch.nn.Module, model_path: str):
    SafetensorsAligner.load_weights(model, model_path)
