# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .config import (
    DEEPSEEK_V4_FLASH_SHAPE,
    DeepSeekV4FlashRuntimeBudget,
    DeepSeekV4FlashShape,
)
from .weight_store import DeepSeekV4FlashWeightStore


class DeepSeekV4FlashForCausalLM(nn.Module):
    """No-weights skeleton for DeepSeek V4 Flash registry integration.

    The first implementation stage only resolves the architecture and binds
    inspect-only GGUF metadata. Real forward execution is added after loader,
    quant, MoE, and compressed attention paths are wired.
    """

    def __init__(
        self,
        config: Any | None = None,
        *,
        shape: DeepSeekV4FlashShape = DEEPSEEK_V4_FLASH_SHAPE,
        weight_store: DeepSeekV4FlashWeightStore | None = None,
        runtime_budget: DeepSeekV4FlashRuntimeBudget | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.shape = shape
        self.weight_store = weight_store
        self.runtime_budget = runtime_budget

    def attach_weight_store(
        self,
        weight_store: DeepSeekV4FlashWeightStore,
        runtime_budget: DeepSeekV4FlashRuntimeBudget,
    ) -> None:
        self.weight_store = weight_store
        self.runtime_budget = runtime_budget

    def close(self) -> None:
        if self.weight_store is not None:
            self.weight_store.close()
            self.weight_store = None

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.weight_store is None:
            raise RuntimeError(
                "DeepSeekV4FlashForCausalLM requires an attached GGUF weight store"
            )
        raise NotImplementedError(
            "DeepSeekV4FlashForCausalLM token execution is not wired yet; "
            "first release work is still in inspect-only/native bring-up."
        )
