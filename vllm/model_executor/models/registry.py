# SPDX-License-Identifier: Apache-2.0
import torch.nn as nn
from typing import Dict, Tuple, Type, Any

class _ModelRegistry:
    def __init__(self):
        self.models: Dict[str, Tuple[str, str]] = {
            "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
            "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
            "QWen2MoeForCausalLM": ("qwen2_moe", "Qwen2MoeForCausalLM"),
            "Qwen2MoeForCausalLM": ("qwen2_moe", "Qwen2MoeForCausalLM"),
            "DeepseekV2ForCausalLM": ("deepseek_v2", "DeepseekV2ForCausalLM"),
            "DeepseekV3ForCausalLM": ("deepseek_v2", "DeepseekV3ForCausalLM"),
            "MixtralForCausalLM": ("mixtral", "MixtralForCausalLM"),
            "KimiLinearForCausalLM": ("kimi_linear", "KimiLinearForCausalLM"),
        }

    def resolve_model_cls(self, architectures: list, model_config: Any) -> Tuple[Type[nn.Module], str]:
        for arch in architectures:
            if arch in self.models:
                mod_name, cls_name = self.models[arch]
                import importlib
                mod = importlib.import_module(f"vllm.model_executor.models.{mod_name}")
                return getattr(mod, cls_name), arch
        raise ValueError(f"Unsupported architectures: {architectures}")

    def is_text_generation_model(self, architectures: list, model_config: Any) -> bool: return True
    def is_multimodal_model(self, architectures: list, model_config: Any) -> bool: return False

ModelRegistry = _ModelRegistry()