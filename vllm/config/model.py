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