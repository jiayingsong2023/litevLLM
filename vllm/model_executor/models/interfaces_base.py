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

def is_text_generation_model(model: Any) -> bool:
    return True 

def get_attn_type(model: Any) -> str:
    return getattr(model, "attn_type", "decoder")
