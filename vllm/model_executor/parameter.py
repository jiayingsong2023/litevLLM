# SPDX-License-Identifier: Apache-2.0
import torch
from torch.nn import Parameter
from typing import Callable, Optional

class BasevLLMParameter(Parameter):
    def __new__(cls, data: torch.Tensor | None = None, **kwargs):
        if data is None: data = torch.empty(0)
        return super().__new__(cls, data=data, requires_grad=False)

    def __init__(self, data: torch.Tensor, weight_loader: Optional[Callable] = None):
        self._weight_loader = weight_loader

    @property
    def weight_loader(self):
        return self._weight_loader or self.default_loader

    def default_loader(self, param, loaded_weight):
        param.data.copy_(loaded_weight)

class ModelWeightParameter(BasevLLMParameter): pass
class PackedvLLMParameter(BasevLLMParameter): pass
class ChannelQuantScaleParameter(BasevLLMParameter): pass
class GroupQuantScaleParameter(BasevLLMParameter): pass
class PerTensorScaleParameter(BasevLLMParameter): pass

__all__ = ["BasevLLMParameter", "ModelWeightParameter", "PackedvLLMParameter", 
           "ChannelQuantScaleParameter", "GroupQuantScaleParameter", "PerTensorScaleParameter"]