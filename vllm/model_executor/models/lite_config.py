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
        # Qwen3.5 / some checkpoints expose explicit head_dim (may differ from hidden/num_heads).
        _hd = getattr(hf_config, "head_dim", None)
        if _hd is None:
            _hd = self.hidden_size // max(1, self.num_attention_heads)
        self.head_dim = int(_hd)
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

        # Qwen3.5 linear-attention (GatedDeltaNet) text config
        self.linear_num_value_heads = getattr(hf_config, "linear_num_value_heads", 32)
        self.linear_num_key_heads = getattr(hf_config, "linear_num_key_heads", 16)
        self.linear_key_head_dim = getattr(hf_config, "linear_key_head_dim", 128)
        self.linear_value_head_dim = getattr(hf_config, "linear_value_head_dim", 128)
        self.linear_conv_kernel_dim = getattr(hf_config, "linear_conv_kernel_dim", 4)
        self.hidden_act = getattr(hf_config, "hidden_act", "silu")

    @classmethod
    def from_model_config(cls, model_config: Any) -> "LiteConfig":
        if hasattr(model_config, "hf_config") and model_config.hf_config is not None:
            return cls(model_config.hf_config)
        return cls(model_config)

    def __repr__(self):
        return f"LiteConfig(hidden={self.hidden_size}, layers={self.num_hidden_layers}, heads={self.num_attention_heads})"
