# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

def set_weight_attrs(weight: torch.Tensor, weight_attrs: Dict[str, Any]) -> None:
    """Set attributes on a weight tensor without overwriting existing ones."""
    for key, value in weight_attrs.items():
        if not hasattr(weight, key):
            setattr(weight, key, value)

def replace_submodule(model: nn.Module, module_name: str, new_module: nn.Module) -> None:
    """Replace a submodule by name."""
    parts = module_name.split(".")
    for part in parts[:-1]:
        model = getattr(model, part)
    setattr(model, parts[-1], new_module)

def maybe_disable_graph_partition(model: nn.Module) -> None:
    """Dummy implementation for LitevLLM compatibility."""
    pass
