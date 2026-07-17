# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod

import torch.nn as nn


class QuantizationConfig(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def init_layer(self, layer: nn.Module):
        pass

    @abstractmethod
    def apply(self, layer: nn.Module, x, *args, **kwargs):
        pass

    @abstractmethod
    def load_weights(self, layer: nn.Module, weights_iter):
        pass
