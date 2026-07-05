# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch

from vllm.model_executor.layers.lite_linear import LiteLinear

from .loader import LoRALoader
from .weights import LoRALayerWeights


class LoRAManager:
    def __init__(self, base_model: torch.nn.Module | None = None) -> None:
        self.base_model = base_model
        self._adapters: dict[str, dict[str, LoRALayerWeights]] = {}

    def bind_to_model(self, base_model: torch.nn.Module | None = None) -> None:
        model = base_model if base_model is not None else self.base_model
        if model is None:
            raise ValueError("base_model is required to bind LoRA weights")
        self.base_model = model
        for name, module in model.named_modules():
            if isinstance(module, LiteLinear):
                module.lora_manager = self
                module.lora_target_name = name or getattr(module, "prefix", "")

    def has_adapter(self, lora_name: str | None) -> bool:
        return bool(lora_name) and str(lora_name) in self._adapters

    def add_adapter_weights(
        self,
        lora_name: str,
        weights: dict[str, LoRALayerWeights],
    ) -> None:
        self._adapters[str(lora_name)] = dict(weights)

    def register_adapter(self, *, lora_name: str, lora_path: str | None) -> None:
        if not lora_path:
            raise ValueError(f"LoRA adapter '{lora_name}' is missing lora_path")
        if self.base_model is None:
            raise ValueError("base_model is required to load LoRA weights")
        weights = LoRALoader(self.base_model).load_adapter(
            lora_name=lora_name,
            lora_path=lora_path,
        )
        self.add_adapter_weights(lora_name, weights)

    def compute_delta(
        self,
        *,
        target_name: str | None,
        x: torch.Tensor,
        lora_mapping: Any,
    ) -> torch.Tensor | None:
        active = self._active_adapter_names(lora_mapping)
        if not active:
            return None
        if len(active) != 1:
            raise ValueError("Phase 1 does not support mixed LoRA batches")
        adapter_name = active[0]
        if target_name is None:
            return None
        weights = self._adapters.get(adapter_name, {}).get(str(target_name))
        if weights is None:
            return None
        x2 = x.reshape(-1, x.shape[-1])
        a = weights.lora_a.to(device=x.device, dtype=x.dtype)
        b = weights.lora_b.to(device=x.device, dtype=x.dtype)
        delta = (x2 @ a) @ b
        delta = delta * weights.scaling
        return delta.reshape(*x.shape[:-1], -1)

    @staticmethod
    def _active_adapter_names(lora_mapping: Any) -> list[str]:
        if lora_mapping is None:
            return []
        if isinstance(lora_mapping, str):
            return [lora_mapping] if lora_mapping else []
        if isinstance(lora_mapping, dict):
            values = lora_mapping.values()
        elif isinstance(lora_mapping, (list, tuple)):
            values = lora_mapping
        else:
            values = [lora_mapping]
        names = sorted({str(item) for item in values if item})
        return names
