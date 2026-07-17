# SPDX-License-Identifier: Apache-2.0
from typing import Any, Protocol, TypeVar, runtime_checkable

import torch

T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class VllmModel(Protocol[T_co]):
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> T_co: ...


@runtime_checkable
class VllmModelForTextGeneration(VllmModel[torch.Tensor], Protocol):
    pass


def is_text_generation_model(model: Any) -> bool:
    return True


def get_attn_type(model: Any) -> str:
    return getattr(model, "attn_type", "decoder")
