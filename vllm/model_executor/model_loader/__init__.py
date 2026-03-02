# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Any
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.models.registry import ModelRegistry

def get_model(vllm_config: VllmConfig) -> nn.Module:
    """LitevLLM: Simplified model loader."""
    # 1. Resolve model class from registry
    architectures = getattr(vllm_config.model_config.hf_config, "architectures", [])
    model_cls, _ = ModelRegistry.resolve_model_cls(architectures, vllm_config.model_config)
    
    # 2. Instantiate model on GPU
    model = model_cls(vllm_config).cuda().half()
    
    return model

def get_tokenizer(model_config: Any, **kwargs):
    """LitevLLM: Unified tokenizer loader via Registry."""
    from vllm.tokenizers.registry import get_tokenizer as registry_get_tokenizer
    return registry_get_tokenizer(model_config, **kwargs)
