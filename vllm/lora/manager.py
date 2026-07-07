# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch

from vllm.model_executor.layers.lite_linear import LiteLinear

from .loader import LoRALoader
from .mapping import LoRAMapping
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
        if target_name is None:
            return None
        mapping = self._adapter_mapping(lora_mapping)
        if len(active) == 1 and not any(item is None for item in mapping):
            return self._compute_adapter_delta(active[0], str(target_name), x)

        if x.ndim >= 3 and len(mapping) == x.shape[0]:
            return self._compute_request_delta(mapping, str(target_name), x)

        x2 = x.reshape(-1, x.shape[-1])
        if len(mapping) == x2.shape[0]:
            return self._compute_token_delta(mapping, str(target_name), x)

        raise ValueError(
            "LoRA mapping length must match batch size or flattened token count"
        )

    def _compute_adapter_delta(
        self,
        adapter_name: str,
        target_name: str,
        x: torch.Tensor,
    ) -> torch.Tensor | None:
        weights = self._adapters.get(adapter_name, {}).get(target_name)
        if weights is None:
            return None
        x2 = x.reshape(-1, x.shape[-1])
        a = weights.lora_a.to(device=x.device, dtype=x.dtype)
        b = weights.lora_b.to(device=x.device, dtype=x.dtype)
        delta = (x2 @ a) @ b
        delta = delta * weights.scaling
        return delta.reshape(*x.shape[:-1], -1)

    def _compute_request_delta(
        self,
        mapping: list[str | None],
        target_name: str,
        x: torch.Tensor,
    ) -> torch.Tensor | None:
        output: torch.Tensor | None = None
        for row_idx, adapter_name in enumerate(mapping):
            if not adapter_name:
                continue
            row_delta = self._compute_adapter_delta(
                adapter_name,
                target_name,
                x[row_idx : row_idx + 1],
            )
            if row_delta is None:
                continue
            if output is None:
                output = torch.zeros(
                    (*x.shape[:-1], row_delta.shape[-1]),
                    device=x.device,
                    dtype=x.dtype,
                )
            output[row_idx : row_idx + 1].copy_(row_delta)
        return output

    def _compute_token_delta(
        self,
        mapping: list[str | None],
        target_name: str,
        x: torch.Tensor,
    ) -> torch.Tensor | None:
        x2 = x.reshape(-1, x.shape[-1])
        output2: torch.Tensor | None = None
        for token_idx, adapter_name in enumerate(mapping):
            if not adapter_name:
                continue
            token_delta = self._compute_adapter_delta(
                adapter_name,
                target_name,
                x2[token_idx : token_idx + 1],
            )
            if token_delta is None:
                continue
            if output2 is None:
                output2 = torch.zeros(
                    (x2.shape[0], token_delta.shape[-1]),
                    device=x.device,
                    dtype=x.dtype,
                )
            output2[token_idx : token_idx + 1].copy_(token_delta.reshape(1, -1))
        if output2 is None:
            return None
        return output2.reshape(*x.shape[:-1], -1)

    @staticmethod
    def _active_adapter_names(lora_mapping: Any) -> list[str]:
        if isinstance(lora_mapping, LoRAMapping):
            return lora_mapping.active_adapter_names
        return sorted(
            {str(item) for item in LoRAManager._adapter_mapping(lora_mapping) if item}
        )

    @staticmethod
    def _adapter_mapping(lora_mapping: Any) -> list[str | None]:
        if lora_mapping is None:
            return []
        if isinstance(lora_mapping, LoRAMapping):
            return list(lora_mapping.adapter_ids)
        if isinstance(lora_mapping, str):
            return [lora_mapping] if lora_mapping else []
        if isinstance(lora_mapping, dict):
            values = lora_mapping.values()
        elif isinstance(lora_mapping, (list, tuple)):
            values = lora_mapping
        else:
            values = [lora_mapping]
        return [str(item) if item else None for item in values]
