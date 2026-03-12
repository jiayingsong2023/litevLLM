# SPDX-License-Identifier: Apache-2.0
import torch
from typing import Any, Dict, Optional

class LiteConfig:
    """
    Standardized configuration adapter for LitevLLM.
    Normalizes different attribute names from HuggingFace/GGUF/AWQ configs
    into a unified schema used by LiteDecoderLayer.
    """
    def __init__(self, hf_config: Any):
        # 1. Base Dimensions
        self.hidden_size = getattr(hf_config, "hidden_size", 4096)
        self.intermediate_size = getattr(hf_config, "intermediate_size", 11008)
        self.num_attention_heads = getattr(hf_config, "num_attention_heads", 32)
        self.num_key_value_heads = getattr(hf_config, "num_key_value_heads", self.num_attention_heads)
        self.num_hidden_layers = getattr(hf_config, "num_hidden_layers", 32)
        self.vocab_size = getattr(hf_config, "vocab_size", 32000)
        
        # 2. Precision & Norm
        self.rms_norm_eps = getattr(hf_config, "rms_norm_eps", 
                            getattr(hf_config, "layer_norm_epsilon", 1e-6))
        
        # 3. Context & RoPE
        self.max_position_embeddings = getattr(hf_config, "max_position_embeddings", 
                                       getattr(hf_config, "max_model_len", 2048))
        self.rope_theta = getattr(hf_config, "rope_theta", 10000.0)
        
        # 4. Multi-modal / Specialized (DeepSeek/MLA)
        self.q_lora_rank = getattr(hf_config, "q_lora_rank", None)
        self.kv_lora_rank = getattr(hf_config, "kv_lora_rank", None)
        self.qk_rope_head_dim = getattr(hf_config, "qk_rope_head_dim", None)
        self.v_head_dim = getattr(hf_config, "v_head_dim", None)

    @classmethod
    def from_model_config(cls, model_config: Any) -> "LiteConfig":
        if hasattr(model_config, "hf_config") and model_config.hf_config is not None:
            return cls(model_config.hf_config)
        return cls(model_config)

    def __repr__(self):
        return f"LiteConfig(hidden={self.hidden_size}, layers={self.num_hidden_layers}, heads={self.num_attention_heads})"
