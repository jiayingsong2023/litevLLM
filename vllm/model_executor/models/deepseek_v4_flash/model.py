# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .config import DEEPSEEK_V4_FLASH_SHAPE, DeepSeekV4FlashShape


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
    ) -> None:
        super().__init__()
        self.config = config
        self.shape = shape

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError(
            "DeepSeekV4FlashForCausalLM forward is not wired yet; "
            "first release work is still in inspect-only/native bring-up."
        )
