# SPDX-License-Identifier: Apache-2.0
from typing import Any

import torch.nn as nn


class _ModelRegistry:
    def __init__(self) -> None:
        self.models: dict[str, tuple[str, str]] = {
            "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
            "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
            "QWen2MoeForCausalLM": ("qwen2_moe", "Qwen2MoeForCausalLM"),
            "MixtralForCausalLM": ("mixtral", "MixtralForCausalLM"),
            "KimiLinearForCausalLM": ("kimi_linear", "KimiLinearForCausalLM"),
            "Qwen3_5ForConditionalGeneration": (
                "qwen3_5",
                "Qwen3_5ForConditionalGeneration",
            ),
            "Qwen3_5MoeForConditionalGeneration": (
                "qwen3_5",
                "Qwen3_5MoeForConditionalGeneration",
            ),
            "Gemma4ForConditionalGeneration": (
                "gemma4",
                "Gemma4ForConditionalGeneration",
            ),
            "Gemma4ForCausalLM": ("gemma4", "Gemma4ForCausalLM"),
            "DeepSeekV4FlashForCausalLM": (
                "deepseek_v4_flash",
                "DeepSeekV4FlashForCausalLM",
            ),
        }

    def _infer_architectures_from_model_config(self, model_config: Any) -> list[str]:
        hf_config = getattr(model_config, "hf_config", None)
        if hf_config is None:
            return []

        candidates: list[str] = []
        model_type = getattr(hf_config, "model_type", None)
        if isinstance(model_type, str) and model_type:
            candidates.append(model_type.lower())

        text_cfg = getattr(hf_config, "text_config", None)
        text_model_type = getattr(text_cfg, "model_type", None)
        if isinstance(text_model_type, str) and text_model_type:
            candidates.append(text_model_type.lower())

        if any(c.startswith("gemma4") for c in candidates):
            return ["Gemma4ForConditionalGeneration"]
        if "qwen3_5_moe_text" in candidates:
            return ["Qwen3_5MoeForConditionalGeneration"]
        if "qwen3_5_text" in candidates or "qwen3_5" in candidates:
            return ["Qwen3_5ForConditionalGeneration"]
        if "deepseek_v4" in candidates or "deepseek4" in candidates:
            return ["DeepSeekV4FlashForCausalLM"]
        return []

    def resolve_model_cls(
        self,
        architectures: list,
        model_config: Any,
    ) -> tuple[type[nn.Module], str]:
        if architectures is None:
            architectures = []
        if len(architectures) == 0:
            architectures = self._infer_architectures_from_model_config(model_config)
        for arch in architectures:
            if arch in self.models:
                mod_name, cls_name = self.models[arch]
                import importlib
                mod = importlib.import_module(f"vllm.model_executor.models.{mod_name}")
                return getattr(mod, cls_name), arch
        raise ValueError(f"Unsupported architectures: {architectures}")

    def is_text_generation_model(
        self,
        architectures: list,
        model_config: Any,
    ) -> bool:
        return True

    def is_multimodal_model(self, architectures: list, model_config: Any) -> bool:
        return False


ModelRegistry = _ModelRegistry()
