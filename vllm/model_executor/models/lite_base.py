# SPDX-License-Identifier: Apache-2.0
from typing import Any

import torch.nn as nn


class LiteDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class LiteModel(nn.Module):
    def __init__(self, vllm_config: Any, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.prefix = prefix
        self.embed_tokens: nn.Embedding | None = None
        self.layers = nn.ModuleList()
        self.norm: nn.Module | None = None

    def forward(self, input_ids, positions, kv_caches, attn_metadata):
        # 1. Embedding Lookup
        if self.embed_tokens is not None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_ids  # Fallback for tests

        # 2. Sequential Layers
        for i, layer in enumerate(self.layers):
            layer_kv_cache = kv_caches[i] if kv_caches else None
            hidden_states = layer(
                hidden_states, positions, layer_kv_cache, attn_metadata
            )

        # 3. Final Norm
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)

        return hidden_states
