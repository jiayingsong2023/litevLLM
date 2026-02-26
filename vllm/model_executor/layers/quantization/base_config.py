# SPDX-License-Identifier: Apache-2.0
import torch.nn as nn
from abc import ABC, abstractmethod

class QuantizationConfig(ABC):
    @abstractmethod
    def get_name(self) -> str: pass
    
    @abstractmethod
    def init_layer(self, layer: nn.Module): pass
    
    @abstractmethod
    def apply(self, layer: nn.Module, x): pass
    
    @abstractmethod
    def load_weights(self, layer: nn.Module, weights_iter): pass