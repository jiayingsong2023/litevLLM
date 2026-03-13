# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class LiteBufferManager:
    """
    Static Buffer Manager for LitevLLM.
    Handles 3D slices for Attention and 2D slices for MLP.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LiteBufferManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def init_pool(self, max_batch: int = 32, max_hidden: int = 8192, device: str = "cuda"):
        if self._initialized: return
        # Pre-allocate large buffers
        self.raw_hidden_a = torch.zeros((max_batch, max_hidden), device=device, dtype=torch.float16)
        self.raw_hidden_b = torch.zeros((max_batch, max_hidden), device=device, dtype=torch.float16)
        self.raw_fused_act = torch.zeros((max_batch, max_hidden * 4), device=device, dtype=torch.float16)
        
        # Touch
        self.raw_hidden_a.zero_(); self.raw_hidden_b.zero_(); self.raw_fused_act.zero_()
        self._initialized = True
        print(f"LiteBufferManager: Static pool ready (BS={max_batch}, H={max_hidden})")

    def get_hidden_3d(self, bs: int, n_h: int, h_d: int):
        """Returns [bs, n_h, h_d] slice."""
        return self.raw_hidden_a[:bs, :n_h * h_d].view(bs, n_h, h_d)

    def get_hidden_2d(self, bs: int, dim: int):
        return self.raw_hidden_b[:bs, :dim]

    def get_fused_act(self, bs: int, dim: int):
        return self.raw_fused_act[:bs, :dim]

def set_weight_attrs(weight: torch.Tensor, weight_attrs: Dict[str, Any]) -> None:
    for key, value in weight_attrs.items():
        if not hasattr(weight, key): setattr(weight, key, value)

def replace_submodule(model: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parts = module_name.split(".")
    for part in parts[:-1]: model = getattr(model, part)
    setattr(model, parts[-1], new_module)

def maybe_disable_graph_partition(model: nn.Module) -> None: pass
