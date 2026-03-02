# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Optional, List, Any, Dict
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

    def forward(self, input_ids, positions, kv_caches, attn_metadata, 
                pixel_values: Optional[torch.Tensor] = None):
        
        # 模拟多模态处理：
        # 如果有图像输入，先通过 Projector 转换并拼接到 hidden_states 中
        # 在端到端测试中，为了性能，我们假设图像已经被预处理为 tokens
        
        hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata)
        return self.lm_head(hidden_states)

    def get_multimodal_embeddings(self, **kwargs):
        # 模拟视觉特征提取
        pixel_values = kwargs.get("pixel_values")
        if pixel_values is not None:
            return self.vision_projector(pixel_values)
        return None
