# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Optional, List, Any, Protocol, runtime_checkable

@runtime_checkable
class SupportsMultiModal(Protocol):
    supports_multimodal: bool = True
    def get_multimodal_embeddings(self, **kwargs) -> torch.Tensor: ...

@runtime_checkable
class SupportsLoRA(Protocol):
    def get_lora_mapping(self) -> Any: ...

@runtime_checkable
class HasInnerState(Protocol):
    def get_inner_state(self) -> Any: ...

@runtime_checkable
class SupportsMRoPE(Protocol):
    def get_mrope_input_positions(self, **kwargs) -> Any: ...

@runtime_checkable
class SupportsPP(Protocol): pass

def is_mixture_of_experts(model: nn.Module) -> bool:
    return any(hasattr(m, "experts") for m in model.modules())

def supports_multimodal(model: nn.Module) -> bool:
    return isinstance(model, SupportsMultiModal)

def supports_lora(model: nn.Module) -> bool:
    return isinstance(model, SupportsLoRA)

def supports_pp(model: nn.Module) -> bool: return isinstance(model, SupportsPP)
def supports_mrope(model: nn.Module) -> bool: return isinstance(model, SupportsMRoPE)
def supports_xdrope(model: nn.Module) -> bool: return hasattr(model, "get_xdrope_input_positions")
def has_inner_state(model: nn.Module) -> bool: return isinstance(model, HasInnerState)
