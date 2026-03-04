# SPDX-License-Identifier: Apache-2.0
import torch
import gc
import os
import json
from typing import List, Optional, Union
from vllm.inputs import PromptInput
from vllm.sampling_params import SamplingParams
from vllm.outputs import RequestOutput
from vllm.model_executor.model_loader import get_model, get_tokenizer
from vllm.model_executor.layers.quantization.gguf import clear_gguf_cache
from transformers import AutoConfig

class LLM:
    """
    Main entrypoint for FastInference.
    Supports Hot Model Switching with Architecture Auto-Fix.
    """
    def __init__(self, model: str, **kwargs):
        self.model_path = model
        self.kwargs = kwargs
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Internal method to handle the actual loading process."""
        print(f">>> Loading model: {self.model_path}")
        
        # 1. Load HF Config with fallback for unknown architectures (Qwen3.5/GLM5)
        try:
            hf_config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        except Exception:
            print(f">>> Transformers fallback: Manual parsing for {self.model_path}")
            config_path = os.path.join(self.model_path, "config.json")
            with open(config_path, "r") as f:
                data = json.load(f)
            
            # Resolve text_config for Qwen3.5 style
            cfg_data = data.get("text_config", data)
            class SimpleConfig:
                def __init__(self, d, archs):
                    self.__dict__.update(d)
                    self.architectures = archs
            hf_config = SimpleConfig(cfg_data, data.get("architectures", []))
        
        # 2. Build LitevLLM-Compatible Config Object
        class LiteVllmConfig:
            def __init__(self, path, hf_cfg):
                self.model_config = type('obj', (object,), {
                    'hf_config': hf_cfg,
                    'dtype': torch.float16,
                    'max_model_len': 4096,
                    'model': path,
                    'get_num_kv_heads': lambda x: getattr(hf_cfg, "num_key_value_heads", getattr(hf_cfg, "num_attention_heads", 1)),
                    'get_head_size': lambda: getattr(hf_cfg, "hidden_size", 4096) // getattr(hf_cfg, "num_attention_heads", 32),
                    'get_num_layers': lambda x: getattr(hf_cfg, "num_hidden_layers", 32),
                })
                self.parallel_config = type('obj', (object,), {'tensor_parallel_size': 1, 'pipeline_parallel_size': 1, 'world_size': 1})
                if any(f.endswith(".gguf") for f in os.listdir(path) if os.path.isdir(path)):
                    from vllm.model_executor.layers.quantization.gguf import GGUFConfig
                    self.quant_config = GGUFConfig()
                else:
                    self.quant_config = None

        v_config = LiteVllmConfig(self.model_path, hf_config)
        
        # 3. Get Model and Tokenizer
        self.model = get_model(v_config)
        self.tokenizer = get_tokenizer(v_config.model_config)
        print(f">>> Model {self.model_path} loaded successfully.")

    def switch_model(self, new_model_path: str, **kwargs):
        """Hot-switches to a new model without restarting the process."""
        print(f"\n--- INITIATING HOT SWITCH: {self.model_path} -> {new_model_path} ---")
        self.model = None
        self.tokenizer = None
        clear_gguf_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print(">>> VRAM Eviction Complete.")
        self.model_path = new_model_path
        self.kwargs.update(kwargs)
        self._load_model()
        print("--- HOT SWITCH SUCCESSFUL ---\n")

    def generate(self, prompts=None, sampling_params=None, **kwargs):
        return []
