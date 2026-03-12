# SPDX-License-Identifier: Apache-2.0
import torch
from typing import Callable, List, Optional

def direct_register_custom_op(op_name: str, op_func: Callable, mutates_args: List[str], fake_impl: Optional[Callable] = None):
    # Simplified registration for LitevLLM
    # In Lite architecture, we can often call functions directly, 
    # but we provide this shim for compatibility with common Op structures.
    pass

def is_hip():
    return torch.version.hip is not None
