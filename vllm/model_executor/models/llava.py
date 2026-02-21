# SPDX-License-Identifier: Apache-2.0
"""Flattened, Single-GPU Llava model optimized for LiteEngine and Triton."""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from transformers import LlavaConfig

from vllm.config import VllmConfig
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.layers.activation import get_act_fn
from .clip import CLIPVisionModel
from .siglip import SiglipVisionModel
from .llama import LlamaModel
from vllm.model_executor.models.utils import AutoWeightsLoader

class LlavaMultiModalProjector(nn.Module):
    def __init__(self, config: LlavaConfig, quant_config=None, prefix=""):
        super().__init__()
        self.linear_1 = LiteLinear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=config.multimodal_projector_bias, quant_config=quant_config, prefix=f"{prefix}.linear_1")
        self.act = get_act_fn(config.projector_hidden_act)
        self.linear_2 = LiteLinear(config.text_config.hidden_size, config.text_config.hidden_size, bias=config.multimodal_projector_bias, quant_config=quant_config, prefix=f"{prefix}.linear_2")

    def forward(self, x):
        return self.linear_2(self.act(self.linear_1(x)))

class LlavaForConditionalGeneration(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix=""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        
        # Initialize components directly
        self.vision_tower = CLIPVisionModel(config.vision_config, quant_config=vllm_config.quant_config, prefix=f"{prefix}.vision_tower")
        self.multi_modal_projector = LlavaMultiModalProjector(config, quant_config=vllm_config.quant_config, prefix=f"{prefix}.multi_modal_projector")
        self.language_model = LlamaModel(vllm_config, prefix=f"{prefix}.language_model")

    def embed_multimodal(self, pixel_values):
        # Flattened multimodal embedding path
        vision_outputs = self.vision_tower(pixel_values)
        return self.multi_modal_projector(vision_outputs)

    def forward(self, input_ids, positions, kv_caches, attn_metadata, **kwargs):
        # input_ids already contains space for multimodal embeddings in vLLM V1
        hidden_states = self.language_model(input_ids, positions, kv_caches, attn_metadata)
        return self.language_model.lm_head(hidden_states)

    def load_weights(self, weights):
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)