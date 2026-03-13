# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class LiteBufferManager:
    """
    Production-grade Static Buffer Manager.
    Pre-allocates based on the LARGEST expected dimension (MLP intermediate size).
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LiteBufferManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def init_pool(self, max_batch: int = 32, max_hidden: int = 8192, max_intermediate: int = 32768, device: str = "cuda"):
        if self._initialized: return
        self.max_batch = max_batch
        
        # Buffer A: Primary (used for Norm outputs and attention results)
        self.raw_a = torch.zeros((max_batch, max_intermediate), device=device, dtype=torch.float16)
        # Buffer B: Secondary (used for parallel MLP Gate/Up projections)
        # Size must handle 2 * intermediate_size for non-overlapping Gate/Up
        self.raw_b = torch.zeros((max_batch, max_intermediate * 2), device=device, dtype=torch.float16)
        
        self.raw_a.zero_(); self.raw_b.zero_()
        self._initialized = True
        print(f"LiteBufferManager: Production pool ready (BS={max_batch}, MaxDim={max_intermediate})")

    def get_slice(self, buffer_id: str, bs: int, dim: int, offset_dim: int = 0):
        base = self.raw_a if buffer_id == "a" else self.raw_b
        return base[:bs, offset_dim : offset_dim + dim]

    def get_slice_3d(self, buffer_id: str, bs: int, n_h: int, h_d: int, offset_dim: int = 0):
        return self.get_slice(buffer_id, bs, n_h * h_d, offset_dim).view(bs, n_h, h_d)

def set_weight_attrs(weight: torch.Tensor, weight_attrs: Dict[str, Any]) -> None:
    for key, value in weight_attrs.items():
        if not hasattr(weight, key): setattr(weight, key, value)

def replace_submodule(model: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parts = module_name.split(".")
    for part in parts[:-1]: model = getattr(model, part)
    setattr(model, parts[-1], new_module)

def maybe_disable_graph_partition(model: nn.Module) -> None: pass
