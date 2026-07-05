# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from vllm.model_executor.layers.lite_linear import LiteLinear

from .weights import LoRALayerWeights


class LoRALoader:
    def __init__(self, base_model: torch.nn.Module) -> None:
        self.base_model = base_model
        self._linear_layers = {
            name: module
            for name, module in base_model.named_modules()
            if isinstance(module, LiteLinear)
        }

    def load_adapter(
        self,
        *,
        lora_name: str,
        lora_path: str,
    ) -> dict[str, LoRALayerWeights]:
        path = Path(lora_path)
        config_path = path / "adapter_config.json"
        weights_path = path / "adapter_model.safetensors"
        if not config_path.is_file():
            raise ValueError(f"missing LoRA adapter_config.json: {config_path}")
        if not weights_path.is_file():
            raise ValueError(f"missing LoRA adapter_model.safetensors: {weights_path}")
        config = json.loads(config_path.read_text())
        rank = int(config.get("r", 0) or 0)
        alpha = config.get("lora_alpha", rank)
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        target_modules = self._target_modules(config.get("target_modules"))
        tensors = load_file(str(weights_path))
        pairs: dict[str, dict[str, torch.Tensor]] = {}
        for key, tensor in tensors.items():
            if key.endswith(".lora_A.weight"):
                layer_name = self._normalize_layer_name(key[: -len(".lora_A.weight")])
                pairs.setdefault(layer_name, {})["a"] = tensor
            elif key.endswith(".lora_B.weight"):
                layer_name = self._normalize_layer_name(key[: -len(".lora_B.weight")])
                pairs.setdefault(layer_name, {})["b"] = tensor
        loaded: dict[str, LoRALayerWeights] = {}
        for layer_name, pair in pairs.items():
            short_name = layer_name.rsplit(".", 1)[-1]
            if target_modules and short_name not in target_modules:
                continue
            layer = self._linear_layers.get(layer_name)
            if layer is None:
                continue
            if "a" not in pair or "b" not in pair:
                raise ValueError(f"incomplete LoRA weights for layer '{layer_name}'")
            a = pair["a"].t().contiguous()
            b = pair["b"].t().contiguous()
            expected_a = (int(layer.input_size), rank)
            expected_b = (rank, int(layer.output_size))
            if tuple(a.shape) != expected_a or tuple(b.shape) != expected_b:
                raise ValueError(
                    f"LoRA shape mismatch for layer '{layer_name}': "
                    f"A={tuple(a.shape)} expected={expected_a}, "
                    f"B={tuple(b.shape)} expected={expected_b}"
                )
            loaded[layer_name] = LoRALayerWeights(
                lora_name=lora_name,
                rank=rank,
                alpha=alpha,
                lora_a=a,
                lora_b=b,
            )
        if not loaded:
            raise ValueError("no matching LoRA layers found")
        return loaded

    @staticmethod
    def _target_modules(value: Any) -> set[str]:
        if value is None:
            return set()
        if isinstance(value, str):
            return {value}
        return {str(item) for item in value}

    @staticmethod
    def _normalize_layer_name(name: str) -> str:
        prefixes = (
            "base_model.model.",
            "base_model.",
            "model.",
        )
        for prefix in prefixes:
            if name.startswith(prefix):
                return name[len(prefix) :]
        return name
