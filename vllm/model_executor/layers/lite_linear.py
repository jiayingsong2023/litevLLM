# SPDX-License-Identifier: Apache-2.0
from typing import Any

import torch
import torch.nn as nn


class LiteLinear(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        quant_config: dict | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.prefix = prefix
        self.quant_config = quant_config

        # Lazy initialization: Parameter exists but is empty until loaded
        self.weight = nn.Parameter(torch.empty(0), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        self.lora_manager: Any | None = None
        self.lora_target_name: str | None = None

        if self.quant_config is not None and hasattr(self.quant_config, "init_layer"):
            self.quant_config.init_layer(self)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        lora_mapping = kwargs.pop("lora_mapping", None)
        if args:
            lora_mapping = args[0]
            args = args[1:]

        # Note: Global LRU Caching is now Legacy. Fused AWQ path is preferred.
        if self.quant_config is not None and hasattr(self.quant_config, "apply"):
            qweight = getattr(self, "qweight", None)
            cached_quant_weight = getattr(self, "_quant_weight", None)
            has_quant_ready = (
                qweight is not None and getattr(qweight, "numel", lambda: 0)() > 1
            ) or cached_quant_weight is not None
            if has_quant_ready:
                base_out = self.quant_config.apply(self, x, *args, **kwargs)
                return self._apply_lora_delta(base_out, x, lora_mapping)

        if self.weight.numel() == 0:
            # Fallback for empty/unloaded weights
            base_out = torch.zeros(
                (*x.shape[:-1], self.output_size), device=x.device, dtype=x.dtype
            )
            return self._apply_lora_delta(base_out, x, lora_mapping)

        base_out = torch.nn.functional.linear(
            x, self.weight, getattr(self, "bias", None)
        )
        return self._apply_lora_delta(base_out, x, lora_mapping)

    def _apply_lora_delta(
        self,
        base_out: torch.Tensor,
        x: torch.Tensor,
        lora_mapping: Any,
    ) -> torch.Tensor:
        if self.lora_manager is None or lora_mapping is None:
            return base_out
        delta = self.lora_manager.compute_delta(
            target_name=self.lora_target_name or self.prefix,
            x=x,
            lora_mapping=lora_mapping,
        )
        if delta is None:
            return base_out
        return base_out + delta.to(device=base_out.device, dtype=base_out.dtype)

    def get_fast_data(self) -> tuple[str, tuple]:
        return "dense", (self.weight, getattr(self, "bias", None))
