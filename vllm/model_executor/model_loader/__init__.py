# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Any
import torch
import torch.nn as nn
import os
from vllm.config import VllmConfig
from vllm.model_executor.models.registry import ModelRegistry

def get_model(vllm_config: VllmConfig) -> nn.Module:
    """LitevLLM: Simplified model loader with Safetensors support."""
    # 1. Resolve model class from registry
    architectures = getattr(vllm_config.model_config.hf_config, "architectures", [])
    model_cls, _ = ModelRegistry.resolve_model_cls(architectures, vllm_config.model_config)
    
    # 2. Instantiate model on GPU (Empty for now)
    model = model_cls(vllm_config).cuda().half()
    
    # 3. Load Real Weights if Safetensors exist
    model_path = vllm_config.model_config.model
    if any(f.endswith(".safetensors") for f in os.listdir(model_path) if os.path.isdir(model_path)):
        from vllm.model_executor.model_loader.safetensors import load_safetensors_weights
        load_safetensors_weights(model, model_path)
    
    return model

def get_tokenizer(model_config: Any, **kwargs):
    """LitevLLM: Unified tokenizer loader via Registry."""
    from vllm.tokenizers.registry import get_tokenizer as registry_get_tokenizer
    return registry_get_tokenizer(model_config, **kwargs)
