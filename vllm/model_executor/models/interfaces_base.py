# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Any, Protocol, runtime_checkable, TypeVar

T_co = TypeVar("T_co", covariant=True)

@runtime_checkable
class VllmModel(Protocol[T_co]):
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> T_co: ...

@runtime_checkable
class VllmModelForTextGeneration(VllmModel[torch.Tensor], Protocol): pass

@runtime_checkable
class VllmModelForPooling(VllmModel[torch.Tensor], Protocol): pass

def is_text_generation_model(model: Any) -> bool:
    return True 

def is_pooling_model(model: Any) -> bool:
    return getattr(model, "is_pooling_model", False)

def get_attn_type(model: Any) -> str:
    return getattr(model, "attn_type", "decoder")

def get_default_seq_pooling_type(model: Any) -> Any: return None
def get_default_tok_pooling_type(model: Any) -> Any: return None
