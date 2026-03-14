# SPDX-License-Identifier: Apache-2.0
import torch
from typing import Any, Optional

class ModelConfig:
    def __init__(self, model: str, tokenizer: str, tokenizer_mode: str = "auto", 
                 trust_remote_code: bool = True, dtype: str = "auto", max_model_len: int = 2048):
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_mode = tokenizer_mode
        self.trust_remote_code = trust_remote_code
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.hf_config: Any = None

    def get_num_layers(self, parallel_config: Any) -> int:
        return getattr(self.hf_config, "num_hidden_layers", 12)

    def get_num_kv_heads(self, parallel_config: Any) -> int:
        return getattr(self.hf_config, "num_key_value_heads", getattr(self.hf_config, "num_attention_heads", 12))

    def get_head_size(self) -> int:
        hidden_size = getattr(self.hf_config, "hidden_size", 768)
        num_heads = getattr(self.hf_config, "num_attention_heads", 12)
        return hidden_size // num_heads

    def get_max_model_len(self) -> int:
        return self.max_model_len