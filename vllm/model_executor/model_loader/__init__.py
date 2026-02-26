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
    
    # 3. Weights loading logic is typically called by the Engine, 
    # but here we return the initialized model.
    return model