# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OpenAIServingModels:
    """Minimal model registry for Lite OpenAI serving."""

    default_model_name: str
    model_config: Any
    lora_requests: dict[str, Any] = field(default_factory=dict)

    def model_name(self) -> str:
        return self.default_model_name

    def available_model_names(self) -> set[str]:
        return {self.default_model_name, *self.lora_requests.keys()}

    def is_supported(self, requested_model: str) -> bool:
        return requested_model in self.available_model_names()
