# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from .ops import rms_norm_reference


@dataclass(frozen=True)
class DeepSeekV4FlashBlockReference:
    layer_idx: int
    hidden_size: int
    attention: Callable[[torch.Tensor, int, Any], torch.Tensor]
    moe: Callable[[torch.Tensor], torch.Tensor]
    attn_norm_weight: torch.Tensor
    ffn_norm_weight: torch.Tensor

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        token_idx: int,
        kv_cache: Any,
    ) -> torch.Tensor:
        if hidden.shape != (self.hidden_size,):
            raise ValueError(
                f"hidden shape must be ({self.hidden_size},); "
                f"got {tuple(hidden.shape)}"
            )
        if self.attn_norm_weight.shape != (self.hidden_size,):
            raise ValueError(
                f"attn_norm_weight shape must be ({self.hidden_size},); "
                f"got {tuple(self.attn_norm_weight.shape)}"
            )
        if self.ffn_norm_weight.shape != (self.hidden_size,):
            raise ValueError(
                f"ffn_norm_weight shape must be ({self.hidden_size},); "
                f"got {tuple(self.ffn_norm_weight.shape)}"
            )

        attn_input = rms_norm_reference(hidden, self.attn_norm_weight)
        attn_output = self.attention(attn_input, token_idx, kv_cache)
        if attn_output.shape != hidden.shape:
            raise ValueError(
                "attention output shape must match hidden shape; "
                f"got {tuple(attn_output.shape)} and {tuple(hidden.shape)}"
            )
        hidden = hidden.to(torch.float32) + attn_output.to(torch.float32)

        ffn_input = rms_norm_reference(hidden, self.ffn_norm_weight)
        ffn_output = self.moe(ffn_input)
        if ffn_output.shape != hidden.shape:
            raise ValueError(
                "MoE output shape must match hidden shape; "
                f"got {tuple(ffn_output.shape)} and {tuple(hidden.shape)}"
            )
        return hidden + ffn_output.to(torch.float32)
