# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from typing import Any

from transformers import PretrainedConfig

from vllm.transformers_utils.configs.speculators.algos import (
    SUPPORTED_SPECULATORS_TYPES,
)

__all__ = ["SpeculatorsConfig"]

class SpeculatorsConfig(PretrainedConfig):
    model_type = "speculators"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        **kwargs,
    ) -> "SpeculatorsConfig":
        Build vLLM-compatible speculative configuration from speculators format.

        This method extracts and transforms speculative configuration from the
        speculators format into the structure expected by vLLM.

        Args:
            config_dict: Configuration dictionary in speculators format

        Returns:
            Dictionary with vLLM-compatible speculative configuration
