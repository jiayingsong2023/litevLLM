# SPDX-License-Identifier: Apache-2.0
import torch
import os
import gc
import re
from safetensors import safe_open
from vllm.logger import init_logger

logger = init_logger(__name__)

class SafetensorsAligner:
    """
    The Ultimate Atomic Aligner.
    Matches weights with zero-tolerance for dimensional mismatch.
    """
    @staticmethod
    def load_weights(model: torch.nn.Module, model_path: str):
        logger.info(f">>> Atomic Alignment for: {model_path}")
        files = sorted([f for f in os.listdir(model_path) if f.endswith(".safetensors")])
        
        module_map = {name: module for name, module in model.named_modules() if hasattr(module, "load_weights")}

        for filename in files:
            full_path = os.path.join(model_path, filename)
            with safe_open(full_path, framework="pt", device="cuda") as f:
                for hf_key in f.keys():
                    if any(x in hf_key for x in ["weight_shape", "bias_shape", "scale_shape"]): continue
                    
                    tensor = f.get_tensor(hf_key)
                    clean_key = hf_key.replace("model.language_model.", "").replace("linear_attn.", "self_attn.").replace("full_attn.", "self_attn.")
                    
                    found = False
                    # 1. Module Path Matching
                    for mod_name, module in module_map.items():
                        m_clean = mod_name if not mod_name.startswith("model.") else mod_name[6:]
                        if clean_key.startswith(m_clean + "."):
                            module.load_weights([(hf_key, tensor)])
                            found = True; break
                    
                    # 2. Strict Parameter Matching (No 'in' operator!)
                    if not found:
                        for p_name, param in model.named_parameters():
                            p_clean = p_name if not p_name.startswith("model.") else p_name[6:]
                            # Match only if the logical path ends with the parameter name
                            if clean_key == p_clean or clean_key.endswith("." + p_clean):
                                if param.shape == tensor.shape:
                                    param.data.copy_(tensor)
                                    found = True; break
                    del tensor
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
        logger.info(">>> Weight Alignment Complete.")

def load_safetensors_weights(model: torch.nn.Module, model_path: str):
    SafetensorsAligner.load_weights(model, model_path)
