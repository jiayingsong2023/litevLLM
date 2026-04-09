# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Optional
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.layers.lite_linear import LiteLinear

class Qwen2VLForCausalLM(nn.Module):
    """
    LitevLLM: Simplified Qwen2-VL simulation for performance benchmarking.
    It adds a Vision Projector to the standard Llama architecture.
    """
    def __init__(self, vllm_config, prefix=""):
        super().__init__()
        # 1. Base Backbone (Llama-like)
        self.model = LlamaModel(vllm_config, prefix=f"{prefix}.model")
        
        # 2. Vision Projector (Simulator)
        # Typically maps vision features (e.g. 1024) to LLM hidden size (e.g. 2048)
        self.vision_projector = LiteLinear(1024, 2048, bias=True, prefix=f"{prefix}.vision_projector")
        
        # 3. LM Head
        self.lm_head = LiteLinear(2048, vllm_config.model_config.hf_config.vocab_size, bias=False)

    def forward(
        self,
        input_ids,
        positions,
        kv_caches,
        attn_metadata,
        pixel_values: Optional[torch.Tensor] = None,
        multimodal_embeddings: Optional[torch.Tensor] = None,
        lora_mapping=None,
    ):
        del pixel_values
        hidden_inputs = self._merge_multimodal_embeddings(
            input_ids=input_ids,
            multimodal_embeddings=multimodal_embeddings,
        )
        hidden_states = self.model(
            hidden_inputs,
            positions,
            kv_caches,
            attn_metadata,
            lora_mapping=lora_mapping,
        )
        return self.lm_head(hidden_states)

    def _merge_multimodal_embeddings(
        self,
        *,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if input_ids.dtype == torch.long:
            hidden_states = self.model.embed_tokens(input_ids)
        else:
            hidden_states = input_ids
        if multimodal_embeddings is None:
            return hidden_states
        if multimodal_embeddings.dim() == 2:
            multimodal_embeddings = multimodal_embeddings.unsqueeze(1)
        multimodal_context = multimodal_embeddings.mean(dim=1, keepdim=True)
        multimodal_context = multimodal_context.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        return hidden_states + multimodal_context.expand(
            -1, hidden_states.shape[1], -1
        )

    def get_multimodal_embeddings(self, **kwargs):
        # 模拟视觉特征提取
        pixel_values = kwargs.get("pixel_values")
        if pixel_values is not None:
            return self.vision_projector(pixel_values)
        return None
