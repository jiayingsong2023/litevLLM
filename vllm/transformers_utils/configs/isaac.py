# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from transformers import Qwen3Config
from transformers.models.siglip2.configuration_siglip2 import Siglip2VisionConfig

class PixelShuffleSiglip2VisionConfig(Siglip2VisionConfig):

    model_type = "pixel_shuffle_siglip2"
    base_config_key = "vision_config"

    def __init__(
        self,
        pixel_shuffle_scale_factor: int = 1,
        num_patches: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Add our custom fields
        self.pixel_shuffle_scale_factor = pixel_shuffle_scale_factor
        self.num_patches = num_patches

class IsaacConfig(Qwen3Config):
