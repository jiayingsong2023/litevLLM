# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LoRALayerWeights:
    lora_name: str
    rank: int
    alpha: int | float
    lora_a: torch.Tensor
    lora_b: torch.Tensor

    @property
    def scaling(self) -> float:
        return float(self.alpha) / float(self.rank)
