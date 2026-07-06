# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.models.lite_config import LiteConfig

from .policy_utils import _get_eps


class Gemma4VisionDenseLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(0), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(0), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.numel() == 0:
            return torch.zeros(
                (*x.shape[:-1], self.out_features),
                dtype=x.dtype,
                device=x.device,
            )
        weight = self.weight.to(device=x.device)
        bias = self.bias
        if bias is not None and bias.numel() > 0:
            bias = bias.to(device=x.device, dtype=weight.dtype)
        return F.linear(x.to(dtype=weight.dtype), weight, bias)


class Gemma4VisionLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = Gemma4VisionDenseLinear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class Gemma4VisionPatchEmbedder(nn.Module):
    def __init__(self, config: LiteConfig) -> None:
        super().__init__()
        self.patch_size = int(getattr(config, "patch_size", 16) or 16)
        patch_dim = self.patch_size * self.patch_size * 3
        self.input_proj = Gemma4VisionDenseLinear(
            patch_dim,
            config.hidden_size,
        )
        self.position_embedding_table = nn.Parameter(torch.empty(0))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.dim() != 4:
            raise ValueError("Gemma4 vision pixel_values must be BCHW")
        bsz, channels, height, width = pixel_values.shape
        if channels != 3:
            raise ValueError("Gemma4 vision pixel_values must have 3 channels")
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError("Gemma4 vision image size must align to patch_size")
        patches = F.unfold(
            pixel_values,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).transpose(1, 2)
        x = self.input_proj(patches)
        if self.position_embedding_table.numel() == 0:
            return x
        pos = self.position_embedding_table[0, : x.shape[1], :]
        return x + pos.to(dtype=x.dtype, device=x.device).unsqueeze(0)


class Gemma4VisionAttention(nn.Module):
    def __init__(self, config: LiteConfig) -> None:
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.head_dim = int(
            getattr(config, "head_dim", config.hidden_size // self.num_heads)
        )
        hidden_size = int(config.hidden_size)
        self.scale = self.head_dim ** -0.5
        self.q_proj = Gemma4VisionLinear(hidden_size, self.num_heads * self.head_dim)
        self.k_proj = Gemma4VisionLinear(hidden_size, self.num_heads * self.head_dim)
        self.v_proj = Gemma4VisionLinear(hidden_size, self.num_heads * self.head_dim)
        self.o_proj = Gemma4VisionLinear(self.num_heads * self.head_dim, hidden_size)
        self.q_norm = RMSNorm(self.head_dim, eps=_get_eps(config))
        self.k_norm = RMSNorm(self.head_dim, eps=_get_eps(config))

    @staticmethod
    def _head_norm(norm: RMSNorm, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        return norm(x.reshape(-1, shape[-1])).view(*shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        q = self._head_norm(self.q_norm, q).transpose(1, 2)
        k = self._head_norm(self.k_norm, k).transpose(1, 2)
        v = v.transpose(1, 2)
        attn = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scale,
        )
        attn = attn.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.o_proj(attn)


class Gemma4VisionMLP(nn.Module):
    def __init__(self, config: LiteConfig) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate_size = int(config.intermediate_size)
        self.gate_proj = Gemma4VisionLinear(hidden_size, intermediate_size)
        self.up_proj = Gemma4VisionLinear(hidden_size, intermediate_size)
        self.down_proj = Gemma4VisionLinear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.gelu(self.gate_proj(x), approximate="tanh")
        return self.down_proj(gate * self.up_proj(x))


class Gemma4VisionEncoderLayer(nn.Module):
    def __init__(self, config: LiteConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=_get_eps(config))
        self.self_attn = Gemma4VisionAttention(config)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=_get_eps(config),
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size,
            eps=_get_eps(config),
        )
        self.mlp = Gemma4VisionMLP(config)
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size,
            eps=_get_eps(config),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.post_attention_layernorm(
            self.self_attn(self.input_layernorm(x))
        )
        x = x + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(x))
        )
        return x


class Gemma4VisionEncoder(nn.Module):
    def __init__(self, config: LiteConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            Gemma4VisionEncoderLayer(config)
            for _ in range(int(config.num_hidden_layers))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class Gemma4VisionTower(nn.Module):
    def __init__(self, hf_vision_config: Any) -> None:
        super().__init__()
        self.config = LiteConfig(hf_vision_config)
        self.patch_embedder = Gemma4VisionPatchEmbedder(self.config)
        self.std_bias = nn.Parameter(torch.zeros(self.config.hidden_size))
        self.std_scale = nn.Parameter(torch.ones(self.config.hidden_size))
        self.encoder = Gemma4VisionEncoder(self.config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedder(pixel_values)
        x = (x + self.std_bias.to(dtype=x.dtype, device=x.device)) * self.std_scale.to(
            dtype=x.dtype,
            device=x.device,
        )
        return self.encoder(x)


class Gemma4VisionProjector(nn.Module):
    def __init__(self, vision_hidden_size: int, text_hidden_size: int) -> None:
        super().__init__()
        self.embedding_projection = LiteLinear(
            vision_hidden_size,
            text_hidden_size,
            bias=False,
            prefix="embedding_projection",
        )

    def forward(self, x: torch.Tensor, lora_mapping: Any = None) -> torch.Tensor:
        return self.embedding_projection(x, lora_mapping)
