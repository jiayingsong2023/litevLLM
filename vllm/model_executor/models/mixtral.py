# SPDX-License-Identifier: Apache-2.0
"""Flattened, Single-GPU Mixtral model optimized for LiteEngine and Triton."""

import torch
import torch.nn as nn
from typing import Iterable, Optional, Tuple, Any
from transformers import MixtralConfig

from vllm.config import VllmConfig
from vllm.model_executor.models.lite_base import LiteForCausalLM, LiteModel, LiteMoEDecoderLayer

class MixtralLiteModel(LiteModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", **kwargs):
        super().__init__(vllm_config=vllm_config, prefix=prefix, **kwargs)
        # All layers are MoE in Mixtral
        self.layers = nn.ModuleList([
            LiteMoEDecoderLayer(vllm_config=vllm_config, prefix=f"{prefix}.layers.{i}", layer_idx=i)
            for i in range(self.config.num_hidden_layers)
        ])

class MixtralForCausalLM(LiteForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", **kwargs):
        super().__init__(vllm_config=vllm_config, prefix=prefix, **kwargs)
        self.model = MixtralLiteModel(vllm_config=vllm_config, prefix=f"{prefix}.model")
        
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Mixtral experts use w1, w2, w3 instead of gate_proj, down_proj, up_proj in checkpoint
        # LiteMoE handles FusedMoE which expects expert_params_mapping
        # For now, use the standard loader. SharedFusedMoE.make_expert_params_mapping 
        # is called inside FusedMoE.load_weights if implemented.
        return super().load_weights(weights)
