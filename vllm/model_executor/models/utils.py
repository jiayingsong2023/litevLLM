# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Iterable, Tuple, Any

def maybe_prefix(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name

class PPMissingLayer(nn.Module):
    """LitevLLM placeholder for missing layers."""
    def forward(self, *args, **kwargs): return args[0] if args else None

def default_weight_loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor):
    """LitevLLM: Basic weight copy from checkpoint to parameter."""
    if param.data.shape != loaded_weight.shape:
        # Handle cases like transposed weights or different layouts
        if param.data.numel() == loaded_weight.numel():
            loaded_weight = loaded_weight.view(param.data.shape)
        else:
            raise ValueError(f"Shape mismatch: {param.data.shape} vs {loaded_weight.shape}")
    param.data.copy_(loaded_weight)

def load_weights_lite(model: nn.Module, weights: Iterable[Tuple[str, torch.Tensor]]):
    """
    Standardized weight loading for LiteModels.
    Iterates through state dict and maps to LiteLinear/RMSNorm.
    """
    params_dict = dict(model.named_parameters())
    for name, loaded_weight in weights:
        if name in params_dict:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
        else:
            # Special handling for fused qkv weights if needed
            pass