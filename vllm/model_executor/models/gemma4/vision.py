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

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor | None = None,
        padding_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        patches, pixel_position_ids = self._patches_and_positions(
            pixel_values,
            pixel_position_ids,
        )
        patches = 2 * (patches - 0.5)
        x = self.input_proj(patches)
        if self.position_embedding_table.numel() == 0:
            return x
        if padding_positions is None:
            padding_positions = (pixel_position_ids == -1).all(dim=-1)
        clamped = pixel_position_ids.clamp(min=0)
        x_pos = F.embedding(clamped[..., 0], self.position_embedding_table[0])
        y_pos = F.embedding(clamped[..., 1], self.position_embedding_table[1])
        pos = x_pos + y_pos
        pos = torch.where(padding_positions.unsqueeze(-1), 0.0, pos)
        return x + pos.to(dtype=x.dtype, device=x.device)

    def _patches_and_positions(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pixel_values.dim() == 3:
            if pixel_position_ids is None:
                raise ValueError("Gemma4 patchified pixels require image_position_ids")
            return pixel_values, pixel_position_ids.to(device=pixel_values.device)
        if pixel_values.dim() != 4:
            raise ValueError("Gemma4 vision pixel_values must be BCHW or BND")
        bsz, channels, height, width = pixel_values.shape
        if channels != 3:
            raise ValueError("Gemma4 vision pixel_values must have 3 channels")
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError("Gemma4 vision image size must align to patch_size")
        patch_rows = height // self.patch_size
        patch_cols = width // self.patch_size
        patches = pixel_values.permute(0, 2, 3, 1).reshape(
            bsz,
            patch_rows,
            self.patch_size,
            patch_cols,
            self.patch_size,
            channels,
        )
        patches = patches.permute(0, 1, 3, 2, 4, 5).reshape(
            bsz,
            patch_rows * patch_cols,
            self.patch_size * self.patch_size * channels,
        )
        if pixel_position_ids is not None:
            return patches, pixel_position_ids.to(device=pixel_values.device)
        grid_x, grid_y = torch.meshgrid(
            torch.arange(patch_cols, device=pixel_values.device),
            torch.arange(patch_rows, device=pixel_values.device),
            indexing="xy",
        )
        positions = torch.stack((grid_x, grid_y), dim=-1).reshape(1, -1, 2)
        return patches, positions.expand(bsz, -1, -1)


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
        rope = getattr(config, "rope_parameters", None) or {}
        self.rope_theta = float(rope.get("rope_theta", 100.0))

    @staticmethod
    def _head_norm(norm: RMSNorm, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        return norm(x.reshape(-1, shape[-1])).view(*shape)

    def _rope_cos_sin(
        self,
        x: torch.Tensor,
        pixel_position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_dim = self.head_dim // 2
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, spatial_dim, 2, device=x.device, dtype=torch.float32)
                / spatial_dim
            )
        )
        cos_parts = []
        sin_parts = []
        positions = pixel_position_ids.clamp(min=0).to(device=x.device)
        for dim in range(2):
            freqs = torch.einsum(
                "bn,d->bnd",
                positions[..., dim].to(torch.float32),
                inv_freq,
            )
            emb = torch.cat((freqs, freqs), dim=-1)
            cos_parts.append(emb.cos().to(dtype=x.dtype))
            sin_parts.append(emb.sin().to(dtype=x.dtype))
        return torch.cat(cos_parts, dim=-1), torch.cat(sin_parts, dim=-1)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        ndim = 2
        rotated = 2 * (x.shape[-1] // (2 * ndim))
        if rotated <= 0:
            return x
        parts = torch.split(x, [rotated, rotated], dim=-1)
        cos_parts = torch.split(cos, [rotated, rotated], dim=-1)
        sin_parts = torch.split(sin, [rotated, rotated], dim=-1)
        out = []
        for part, c, s in zip(parts, cos_parts, sin_parts):
            c = c.unsqueeze(2)
            s = s.unsqueeze(2)
            out.append((part * c) + (self._rotate_half(part) * s))
        return torch.cat(out, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        q = self._head_norm(self.q_norm, q)
        k = self._head_norm(self.k_norm, k)
        cos, sin = self._rope_cos_sin(q, pixel_position_ids)
        q = self._apply_rope(q, cos, sin).transpose(1, 2)
        k = self._apply_rope(k, cos, sin).transpose(1, 2)
        v = v.transpose(1, 2)
        sdpa_mask = None
        if attention_mask is not None:
            sdpa_mask = attention_mask[:, None, None, :].to(device=x.device)
        attn = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=sdpa_mask,
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

    def forward(
        self,
        x: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.post_attention_layernorm(
            self.self_attn(
                self.input_layernorm(x),
                pixel_position_ids,
                attention_mask,
            )
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

    def forward(
        self,
        x: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, pixel_position_ids, attention_mask)
        return x


class Gemma4VisionTower(nn.Module):
    def __init__(self, hf_vision_config: Any) -> None:
        super().__init__()
        self.config = LiteConfig(hf_vision_config)
        self.patch_embedder = Gemma4VisionPatchEmbedder(self.config)
        self.std_bias = nn.Parameter(torch.zeros(self.config.hidden_size))
        self.std_scale = nn.Parameter(torch.ones(self.config.hidden_size))
        self.encoder = Gemma4VisionEncoder(self.config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if image_position_ids is None and pixel_values.dim() == 4:
            _, _, height, width = pixel_values.shape
            patch_rows = height // int(self.config.patch_size)
            patch_cols = width // int(self.config.patch_size)
            grid_x, grid_y = torch.meshgrid(
                torch.arange(patch_cols, device=pixel_values.device),
                torch.arange(patch_rows, device=pixel_values.device),
                indexing="xy",
            )
            image_position_ids = torch.stack((grid_x, grid_y), dim=-1).reshape(
                1,
                -1,
                2,
            )
            image_position_ids = image_position_ids.expand(
                pixel_values.shape[0],
                -1,
                -1,
            )
        if image_position_ids is None:
            raise ValueError("Gemma4 vision requires image_position_ids")
        padding = (image_position_ids == -1).all(dim=-1).to(device=pixel_values.device)
        x = self.patch_embedder(pixel_values, image_position_ids, padding)
        x = self.encoder(x, image_position_ids, ~padding)
        pool = int(getattr(self.config, "pooling_kernel_size", 1) or 1)
        output_length = x.shape[1] // (pool * pool)
        x = x.masked_fill(padding.unsqueeze(-1), 0.0)
        if x.shape[1] != output_length:
            x, valid = self._pool_by_positions(
                x,
                image_position_ids,
                padding,
                output_length,
            )
        else:
            valid = ~padding
        x = x.float() * (float(self.config.hidden_size) ** 0.5)
        x = (x - self.std_bias.float().to(device=x.device)) * self.std_scale.float().to(
            device=x.device
        )
        x = x.to(dtype=pixel_values.dtype)
        return x[valid]

    def _pool_by_positions(
        self,
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
        output_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_len = hidden_states.shape[1]
        pool = int((input_len // output_length) ** 0.5)
        if pool * pool * output_length != input_len:
            raise ValueError("Gemma4 vision pooling shape mismatch")
        positions = pixel_position_ids.clamp(min=0)
        pooled_positions = positions // pool
        max_x = int(pooled_positions[..., 0].max().item()) + 1
        pooled_index = pooled_positions[..., 1] * max_x + pooled_positions[..., 0]
        pooled_index = pooled_index.masked_fill(padding_positions, 0)
        weights = torch.nn.functional.one_hot(
            pooled_index,
            num_classes=output_length,
        ).to(dtype=hidden_states.dtype, device=hidden_states.device)
        weights = weights.masked_fill(padding_positions.unsqueeze(-1), 0.0)
        pooled = torch.einsum("bnh,bnl->blh", hidden_states, weights) / float(
            pool * pool
        )
        valid = weights.sum(dim=1) > 0
        return pooled, valid


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
