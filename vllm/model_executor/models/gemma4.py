# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe
from vllm.model_executor.layers.quantization.tensor import (
    dequantize_awq_pytorch,
    dequantize_symmetric_packed_int4_pytorch,
)
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

from .lite_config import LiteConfig


def _get_eps(config: LiteConfig) -> float:
    return float(
        getattr(config, "rms_norm_eps", getattr(config, "layer_norm_epsilon", 1e-6))
    )


def _meta_get(meta: Any, key: str, default: Any = None) -> Any:
    if isinstance(meta, dict):
        return meta.get(key, default)
    return getattr(meta, key, default)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _repeat_kv_for_gqa(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    bsz, n_kv, seqlen, hd = x.shape
    x = x[:, :, None, :, :].expand(bsz, n_kv, n_rep, seqlen, hd)
    return x.reshape(bsz, n_kv * n_rep, seqlen, hd)


def _causal_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    local_window: Optional[int] = None,
    softcap: Optional[float] = None,
) -> torch.Tensor:
    n_rep = q.shape[1] // k.shape[1]
    k_full = _repeat_kv_for_gqa(k, n_rep)
    v_full = _repeat_kv_for_gqa(v, n_rep)
    scores = torch.matmul(q, k_full.transpose(2, 3)) * scale
    q_len = int(q.shape[2])
    kv_len = int(k_full.shape[2])
    # Support both square (q_len == kv_len) and chunked prefill (q_len < kv_len).
    # For chunked prefill, q is assumed to be the tail segment of the kv timeline.
    q_pos = torch.arange(q_len, device=q.device, dtype=torch.long) + (kv_len - q_len)
    k_pos = torch.arange(kv_len, device=q.device, dtype=torch.long)
    causal = k_pos[None, :] > q_pos[:, None]
    scores = scores.masked_fill(causal[None, None, :, :], float("-inf"))
    if local_window is not None and local_window > 0:
        dist = q_pos[:, None] - k_pos[None, :]
        local_mask = dist >= int(local_window)
        scores = scores.masked_fill(local_mask[None, None, :, :], float("-inf"))
    if softcap is not None and softcap > 0:
        scores = torch.tanh(scores / softcap) * softcap
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(probs, v_full)
    return out.transpose(1, 2).contiguous()


def _decode_int4_row(
    cache: torch.Tensor,
    scale_cache: Optional[torch.Tensor],
    block_idx: int,
    block_offset: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    packed = cache[block_idx, block_offset, :num_kv_heads, : head_dim // 2].to(torch.int32)
    low = ((packed << 28) >> 28).to(torch.float32)
    high = ((packed << 24) >> 28).to(torch.float32)
    out = torch.empty((num_kv_heads, head_dim), device=cache.device, dtype=torch.float32)
    out[:, : head_dim // 2] = low
    out[:, head_dim // 2 :] = high
    if scale_cache is not None:
        scale = scale_cache[block_idx, block_offset, :num_kv_heads, 0].to(torch.float32)
        out = out * scale[:, None]
    return out


def _decode_int4_rows(
    packed_rows: torch.Tensor,
    scales: Optional[torch.Tensor],
    head_dim: int,
) -> torch.Tensor:
    # packed_rows: [T, num_kv_heads, head_dim/2] uint8/int32
    packed_i32 = packed_rows.to(torch.int32)
    low = ((packed_i32 << 28) >> 28).to(torch.float32)
    high = ((packed_i32 << 24) >> 28).to(torch.float32)
    out = torch.empty(
        (packed_rows.shape[0], packed_rows.shape[1], head_dim),
        device=packed_rows.device,
        dtype=torch.float32,
    )
    half = head_dim // 2
    out[:, :, :half] = low
    out[:, :, half:] = high
    if scales is not None:
        out = out * scales.to(torch.float32).unsqueeze(-1)
    return out


def _gather_recent_kv(
    kv_cache: Any,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    batch_idx: int,
    num_kv_heads: int,
    head_dim: int,
    local_window: Optional[int],
    kv_cache_dtype: str,
    kv_scale_cache: Optional[tuple[Any, Any]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    k_cache, v_cache = kv_cache
    block_size = int(k_cache.shape[1])
    seq_len = int(seq_lens[batch_idx].item())
    if local_window is None or int(local_window) <= 0:
        start = 0
    else:
        start = max(0, seq_len - int(local_window))
    k_scale_cache = kv_scale_cache[0] if kv_scale_cache is not None else None
    v_scale_cache = kv_scale_cache[1] if kv_scale_cache is not None else None
    is_int4 = "int4" in str(kv_cache_dtype).lower()
    token_positions = torch.arange(start, seq_len, device=block_tables.device, dtype=torch.long)
    bt_row = block_tables[batch_idx]
    block_indices = bt_row[token_positions // block_size]
    block_offsets = torch.remainder(token_positions, block_size)

    if is_int4:
        k_packed = k_cache[block_indices, block_offsets, :num_kv_heads, : head_dim // 2]
        v_packed = v_cache[block_indices, block_offsets, :num_kv_heads, : head_dim // 2]
        k_scales = (
            k_scale_cache[block_indices, block_offsets, :num_kv_heads, 0]
            if k_scale_cache is not None
            else None
        )
        v_scales = (
            v_scale_cache[block_indices, block_offsets, :num_kv_heads, 0]
            if v_scale_cache is not None
            else None
        )
        k = _decode_int4_rows(k_packed, k_scales, head_dim).unsqueeze(0)
        v = _decode_int4_rows(v_packed, v_scales, head_dim).unsqueeze(0)
    else:
        k = (
            k_cache[block_indices, block_offsets, :num_kv_heads, :head_dim]
            .to(torch.float32)
            .unsqueeze(0)
        )
        v = (
            v_cache[block_indices, block_offsets, :num_kv_heads, :head_dim]
            .to(torch.float32)
            .unsqueeze(0)
        )
    return k, v


def _should_use_full_decode_reference(kv_cache_dtype: str) -> bool:
    kvt = str(kv_cache_dtype).lower()
    # Stability-first path for Gemma4 decode alignment:
    # when KV is full precision, use the same reference attention math as no-cache path.
    return ("int4" not in kvt) and ("fp8" not in kvt)


def _write_full_precision_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    flat_k = k.reshape(-1, num_kv_heads, head_dim).contiguous()
    flat_v = v.reshape(-1, num_kv_heads, head_dim).contiguous()
    valid = slot_mapping >= 0
    if not bool(valid.any()):
        return
    slots = slot_mapping[valid].to(torch.long)
    block_size = int(k_cache.shape[1])
    block_idx = torch.div(slots, block_size, rounding_mode="floor")
    block_off = torch.remainder(slots, block_size)
    k_cache[block_idx, block_off, :num_kv_heads, :head_dim] = flat_k[valid].to(
        dtype=k_cache.dtype
    )
    v_cache[block_idx, block_off, :num_kv_heads, :head_dim] = flat_v[valid].to(
        dtype=v_cache.dtype
    )


def _layer_type_for_idx(config: LiteConfig, layer_idx: int) -> str:
    layer_types = getattr(config, "layer_types", None)
    if isinstance(layer_types, list) and layer_idx < len(layer_types):
        return str(layer_types[layer_idx]).lower()
    return "global"


def _is_local_layer(layer_type: str) -> bool:
    return any(x in layer_type for x in ("local", "sliding"))


class Gemma4LayerRotaryEmbedding(nn.Module):
    def __init__(self, config: LiteConfig, head_size: int, layer_type: str):
        super().__init__()
        self.head_size = int(head_size)
        self.layer_type = layer_type
        self.max_position_embeddings = int(config.max_position_embeddings)
        self.apply_rotary_emb = ApplyRotaryEmb(is_neox_style=True)

        rope_params = {}
        cfg_rope = getattr(config, "rope_parameters", None)
        if isinstance(cfg_rope, dict):
            layer_rope = cfg_rope.get(layer_type)
            if isinstance(layer_rope, dict):
                rope_params = layer_rope
        self.base = float(
            rope_params.get("rope_theta", getattr(config, "rope_theta", 10000.0))
        )
        self.rope_type = str(rope_params.get("rope_type", "default"))
        self.partial_rotary_factor = float(rope_params.get("partial_rotary_factor", 1.0))
        inv_freq = self._build_inv_freq()
        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def _build_inv_freq(self) -> torch.Tensor:
        if self.rope_type == "proportional":
            rope_angles = int(self.partial_rotary_factor * self.head_size // 2)
            rope_angles = max(0, min(self.head_size // 2, rope_angles))
            inv_rot = 1.0 / (
                self.base
                ** (
                    torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32)
                    / float(self.head_size)
                )
            )
            no_rot = self.head_size // 2 - rope_angles
            if no_rot > 0:
                inv_freq = torch.cat(
                    (inv_rot, torch.zeros(no_rot, dtype=torch.float32)), dim=0
                )
            else:
                inv_freq = inv_rot
            return inv_freq
        return 1.0 / (
            self.base
            ** (
                torch.arange(0, self.head_size, 2, dtype=torch.float32)
                / float(self.head_size)
            )
        )

    def forward(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if positions.device != self.cos_cached.device:
            positions = positions.to(self.cos_cached.device)
        cos = self.cos_cached[positions]
        sin = self.sin_cached[positions]
        return (
            self.apply_rotary_emb.forward_native(query, cos, sin),
            self.apply_rotary_emb.forward_native(key, cos, sin),
        )


def _get_rope(config: LiteConfig, head_size: int, layer_type: str):
    rope_params = {}
    cfg_rope = getattr(config, "rope_parameters", None)
    if isinstance(cfg_rope, dict):
        layer_rope = cfg_rope.get(layer_type)
        if isinstance(layer_rope, dict):
            rope_params = layer_rope
    return Gemma4LayerRotaryEmbedding(config, head_size, layer_type)


class Gemma4Attention(nn.Module):
    def __init__(self, config: LiteConfig, quant_config: Any, prefix: str, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = _layer_type_for_idx(config, layer_idx)
        self.is_sliding = _is_local_layer(self.layer_type)
        self.num_heads = int(config.num_attention_heads)
        self.head_dim = int(
            config.head_dim
            if self.is_sliding
            else getattr(config, "global_head_dim", config.head_dim)
        )
        self.use_alternative_attention = bool(
            getattr(config, "attention_k_eq_v", False) and not self.is_sliding
        )
        self.num_kv_heads = int(
            config.num_key_value_heads
            if self.is_sliding
            else getattr(config, "num_global_key_value_heads", config.num_key_value_heads)
        )
        # HF Gemma4TextAttention uses q/k RMSNorm and sets attention scaling to 1.0.
        # Keeping it aligned avoids over-damping logits (especially in MoE variants).
        self.scale = 1.0
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.q_proj = LiteLinear(
            config.hidden_size,
            self.q_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn.q_proj",
        )
        self.k_proj = LiteLinear(
            config.hidden_size,
            self.kv_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn.k_proj",
        )
        self.v_proj = None
        if not self.use_alternative_attention:
            self.v_proj = LiteLinear(
                config.hidden_size,
                self.kv_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn.v_proj",
            )
        self.o_proj = LiteLinear(
            self.q_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn.o_proj",
        )
        self.q_norm = RMSNorm(self.head_dim, eps=_get_eps(config))
        self.k_norm = RMSNorm(self.head_dim, eps=_get_eps(config))
        self.v_norm_eps = _get_eps(config)
        self.rotary_emb = _get_rope(config, self.head_dim, self.layer_type)

    @staticmethod
    def _apply_head_norm(norm: RMSNorm, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        y = norm(x.reshape(-1, shape[-1]))
        return y.view(*shape)

    @staticmethod
    def _apply_head_norm_noscale(x: torch.Tensor, eps: float) -> torch.Tensor:
        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        out = x_fp32 * torch.rsqrt(variance + eps)
        return out.to(input_dtype)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: Any,
        attn_metadata: Any,
        lora_mapping: Any = None,
    ) -> torch.Tensor:
        q = self.q_proj(x, lora_mapping)
        k = self.k_proj(x, lora_mapping)
        v = self.v_proj(x, lora_mapping) if self.v_proj is not None else k
        bsz, seqlen = x.shape[:2]
        q = q.view(bsz, seqlen, self.num_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = self._apply_head_norm(self.q_norm, q)
        k = self._apply_head_norm(self.k_norm, k)
        v = self._apply_head_norm_noscale(v, self.v_norm_eps)
        q, k = self.rotary_emb(positions, q, k)
        is_local = self.is_sliding
        local_window = (
            int(getattr(self.config, "sliding_window", 0) or 0) if is_local else None
        )
        softcap = getattr(self.config, "attn_logit_softcapping", None)
        slot_mapping = _meta_get(attn_metadata, "slot_mapping", None)
        if slot_mapping is not None and kv_cache is not None:
            from vllm.kernels.triton.reshape_and_cache import reshape_and_cache

            inf_config = _meta_get(attn_metadata, "config", None)
            kv_cache_dtype = (
                inf_config.kv_type
                if inf_config is not None
                else _meta_get(attn_metadata, "kv_cache_dtype", "auto")
            )
            k_scale = (
                inf_config.k_scale
                if inf_config is not None
                else _meta_get(attn_metadata, "k_scale", 1.0)
            )
            v_scale = (
                inf_config.v_scale
                if inf_config is not None
                else _meta_get(attn_metadata, "v_scale", 1.0)
            )
            k_cache, v_cache = kv_cache
            kv_scale_cache = _meta_get(attn_metadata, "kv_scale_cache", None)
            if kv_scale_cache is not None:
                k_scale_cache, v_scale_cache = kv_scale_cache[self.layer_idx]
            else:
                k_scale_cache, v_scale_cache = (None, None)

            if _should_use_full_decode_reference(str(kv_cache_dtype)):
                _write_full_precision_kv_cache(
                    k,
                    v,
                    k_cache,
                    v_cache,
                    slot_mapping,
                    self.num_kv_heads,
                    self.head_dim,
                )
            else:
                reshape_and_cache(
                    k.reshape(-1, self.num_kv_heads, self.head_dim).contiguous(),
                    v.reshape(-1, self.num_kv_heads, self.head_dim).contiguous(),
                    k_cache,
                    v_cache,
                    slot_mapping,
                    kv_cache_dtype,
                    k_scale,
                    v_scale,
                    k_scale_cache=k_scale_cache,
                    v_scale_cache=v_scale_cache,
                )

            is_prefill = bool(_meta_get(attn_metadata, "is_prefill", False))
            if is_local and is_prefill and seqlen > 1:
                out = _causal_attention_ref(
                    q.transpose(1, 2).float(),
                    k.transpose(1, 2).float(),
                    v.transpose(1, 2).float(),
                    self.scale,
                    local_window=local_window,
                    softcap=softcap,
                ).to(q.dtype).view(bsz, seqlen, -1)
            elif is_local and not is_prefill:
                block_tables = _meta_get(attn_metadata, "block_tables", None)
                seq_lens = _meta_get(attn_metadata, "seq_lens", None)
                kv_dtype_name = (
                    inf_config.kv_type
                    if inf_config is not None
                    else _meta_get(attn_metadata, "kv_cache_dtype", "auto")
                )
                outs = []
                for bi in range(bsz):
                    ctx_window = int(local_window or 0)
                    k_ctx, v_ctx = _gather_recent_kv(
                        kv_cache=kv_cache,
                        block_tables=block_tables,
                        seq_lens=seq_lens,
                        batch_idx=bi,
                        num_kv_heads=self.num_kv_heads,
                        head_dim=self.head_dim,
                        local_window=ctx_window,
                        kv_cache_dtype=str(kv_dtype_name),
                        kv_scale_cache=(k_scale_cache, v_scale_cache),
                    )
                    q_i = q[bi : bi + 1].transpose(1, 2).float()
                    out_i = _causal_attention_ref(
                        q_i,
                        k_ctx.transpose(1, 2).float(),
                        v_ctx.transpose(1, 2).float(),
                        self.scale,
                        local_window=None,
                        softcap=softcap,
                    ).to(q.dtype).view(1, seqlen, -1)
                    outs.append(out_i)
                out = torch.cat(outs, dim=0)
            else:
                from vllm.engine.lite_engine import expand_metadata_for_paged_attention
                from vllm.kernels.triton.paged_attention import paged_attention_v1

                attn_out = torch.empty(
                    (bsz * seqlen, self.num_heads, self.head_dim),
                    device=q.device,
                    dtype=q.dtype,
                )
                block_tables = _meta_get(attn_metadata, "block_tables", None)
                seq_lens = _meta_get(attn_metadata, "seq_lens", None)
                use_full_ref = _should_use_full_decode_reference(str(kv_cache_dtype))
                if use_full_ref and block_tables is not None and seq_lens is not None:
                    outs = []
                    for bi in range(bsz):
                        k_ctx, v_ctx = _gather_recent_kv(
                            kv_cache=kv_cache,
                            block_tables=block_tables,
                            seq_lens=seq_lens,
                            batch_idx=bi,
                            num_kv_heads=self.num_kv_heads,
                            head_dim=self.head_dim,
                            local_window=None,
                            kv_cache_dtype=str(kv_cache_dtype),
                            kv_scale_cache=(k_scale_cache, v_scale_cache),
                        )
                        q_i = q[bi : bi + 1].transpose(1, 2).float()
                        out_i = _causal_attention_ref(
                            q_i,
                            k_ctx.transpose(1, 2).float(),
                            v_ctx.transpose(1, 2).float(),
                            self.scale,
                            local_window=None,
                            softcap=softcap,
                        ).to(q.dtype).view(1, seqlen, -1)
                        outs.append(out_i)
                    out = torch.cat(outs, dim=0)
                    return self.o_proj(out, lora_mapping)
                seq_lens_ext, block_tables_ext = expand_metadata_for_paged_attention(
                    bsz, seqlen, is_prefill, seq_lens, block_tables, q.device
                )
                max_ctx = int(
                    max(
                        self.num_heads * self.head_dim,
                        getattr(self.config, "max_position_embeddings", 4096),
                    )
                )
                paged_attention_v1(
                    attn_out,
                    q.reshape(bsz * seqlen, self.num_heads, self.head_dim).contiguous(),
                    k_cache,
                    v_cache,
                    self.num_heads,
                    self.scale,
                    block_tables_ext,
                    seq_lens_ext,
                    k_cache.shape[1],
                    max_ctx,
                    None,
                    kv_cache_dtype,
                    k_scale,
                    v_scale,
                    k_scale_ptrs=k_scale_cache,
                    v_scale_ptrs=v_scale_cache,
                    num_kv_heads=self.num_kv_heads,
                )
                out = attn_out.view(bsz, seqlen, -1)
        else:
            out = _causal_attention_ref(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                self.scale,
                local_window=local_window,
                softcap=softcap,
            ).view(bsz, seqlen, -1)
        return self.o_proj(out, lora_mapping)


class Gemma4MLP(nn.Module):
    def __init__(self, config: LiteConfig, quant_config: Any, prefix: str):
        super().__init__()
        self.hidden_act = str(
            getattr(config, "hidden_activation", getattr(config, "hidden_act", "silu"))
        ).lower()
        self.gate_proj = LiteLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp.gate_proj",
        )
        self.up_proj = LiteLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp.up_proj",
        )
        self.down_proj = LiteLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp.down_proj",
        )

    def forward(self, x: torch.Tensor, lora_mapping: Any = None) -> torch.Tensor:
        gate = self.gate_proj(x, lora_mapping)
        up = self.up_proj(x, lora_mapping)
        if self.hidden_act in ("gelu", "gelu_pytorch_tanh"):
            act = F.gelu(gate, approximate="tanh")
        else:
            act = F.silu(gate)
        return self.down_proj(act * up, lora_mapping)


def _is_gemma4_moe_enabled(config: LiteConfig) -> bool:
    return bool(
        int(getattr(config, "num_experts", 0) or 0) > 0
        and int(getattr(config, "num_experts_per_tok", 0) or 0) > 0
        and int(getattr(config, "moe_intermediate_size", 0) or 0) > 0
    )


def _is_gemma4_moe_layer(config: LiteConfig, layer_idx: int) -> bool:
    if not _is_gemma4_moe_enabled(config):
        return False
    if hasattr(config, "is_moe_layer"):
        try:
            return bool(config.is_moe_layer(layer_idx))
        except Exception:
            pass
    return True


def _is_gemma4_26b_a4b_like(config: LiteConfig) -> bool:
    return bool(
        int(getattr(config, "hidden_size", 0) or 0) == 2816
        and int(getattr(config, "num_hidden_layers", 0) or 0) == 30
        and int(getattr(config, "num_experts", 0) or 0) >= 64
        and int(getattr(config, "moe_intermediate_size", 0) or 0) > 0
    )


def _residual_add_fp32(residual: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    return (residual.float() + update.float()).to(residual.dtype)


def _reshape_hidden_to_2d(
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    bsz, seqlen, hidden_dim = hidden_states.shape
    return hidden_states.reshape(bsz * seqlen, hidden_dim), (bsz, seqlen, hidden_dim)


def _restore_hidden_from_2d(
    hidden_states_2d: torch.Tensor,
    shape: tuple[int, int, int],
) -> torch.Tensor:
    bsz, seqlen, hidden_dim = shape
    return hidden_states_2d.reshape(bsz, seqlen, hidden_dim)


class Gemma4TopKRouterLite(nn.Module):
    def __init__(self, config: LiteConfig, quant_config: Any, prefix: str):
        super().__init__()
        self.num_experts = int(config.num_experts)
        self.top_k = int(config.num_experts_per_tok)
        self.hidden_size = int(config.hidden_size)
        self.eps = float(_get_eps(config))
        self.scalar_root_size = float(max(1, self.hidden_size)) ** -0.5
        self.proj = LiteLinear(
            config.hidden_size,
            self.num_experts,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.router.proj",
        )
        # Optional router scaling tensors used by Gemma4-26B A4B checkpoints.
        self.scale = nn.Parameter(torch.empty(0), requires_grad=False)
        self.per_expert_scale = nn.Parameter(torch.empty(0), requires_grad=False)

    def forward(
        self,
        hidden_states_2d: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Match HF Gemma4 router math:
        # x = RMSNorm(with_scale=False)(x) * scale * (hidden_size ** -0.5)
        x_fp32 = hidden_states_2d.to(torch.float32)
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x = (x_fp32 * torch.rsqrt(variance + self.eps)).to(hidden_states_2d.dtype)
        if self.scale.numel() > 1:
            x = x * self.scale.to(device=x.device, dtype=x.dtype)
        x = x * self.scalar_root_size
        router_logits = self.proj(x)
        routing_weights = F.softmax(router_logits.float(), dim=-1)
        routing_weights, selected_experts = torch.topk(
            routing_weights,
            k=self.top_k,
            dim=-1,
        )
        routing_weights = routing_weights / routing_weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-8)
        if self.per_expert_scale.numel() > 1:
            per_exp = self.per_expert_scale.to(
                device=routing_weights.device,
                dtype=routing_weights.dtype,
            )
            routing_weights = routing_weights * per_exp[selected_experts]
        return (
            router_logits,
            routing_weights.to(hidden_states_2d.dtype),
            selected_experts,
        )


def _materialize_litelinear_dense_weight_awqaware(
    layer: LiteLinear,
    *,
    out_features: int,
    in_features: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    dense_weight = getattr(layer, "weight", None)
    if isinstance(dense_weight, torch.Tensor) and dense_weight.numel() > 1:
        dense_weight = dense_weight[:out_features, :in_features].contiguous()
        return dense_weight.to(device=device, dtype=dtype)

    qweight = getattr(layer, "qweight", None)
    scales = getattr(layer, "scales", None)
    qzeros = getattr(layer, "qzeros", None)
    group_size = int(getattr(layer, "group_size", 128))
    if qweight is None or not isinstance(qweight, torch.Tensor) or qweight.numel() <= 1:
        raise RuntimeError(
            f"Layer '{getattr(layer, 'prefix', '<unknown>')}' has neither dense nor packed weights."
        )
    if scales is None or not isinstance(scales, torch.Tensor) or scales.numel() <= 1:
        raise RuntimeError(
            f"Layer '{getattr(layer, 'prefix', '<unknown>')}' is missing AWQ scales."
        )

    if isinstance(qzeros, torch.Tensor) and qzeros.numel() > 1:
        dense_weight = dequantize_awq_pytorch(
            qweight.to(device=device, dtype=torch.int32),
            scales.to(device=device),
            qzeros.to(device=device, dtype=torch.int32),
            group_size=group_size,
        )
    else:
        dense_weight = dequantize_symmetric_packed_int4_pytorch(
            qweight.to(device=device, dtype=torch.int32),
            scales.to(device=device),
            group_size=group_size,
        )

    if dense_weight.shape[0] < out_features or dense_weight.shape[1] < in_features:
        raise RuntimeError(
            f"Layer '{getattr(layer, 'prefix', '<unknown>')}' dequantized weight too small: "
            f"got {tuple(dense_weight.shape)}, need ({out_features}, {in_features})"
        )
    return dense_weight[:out_features, :in_features].contiguous().to(device=device, dtype=dtype)


class Gemma4MoeExpertsLite(nn.Module):
    def __init__(self, config: LiteConfig, quant_config: Any, prefix: str):
        super().__init__()
        self.hidden_dim = int(config.hidden_size)
        self.intermediate_dim = int(config.moe_intermediate_size)
        self.num_experts = int(config.num_experts)
        self.top_k = int(config.num_experts_per_tok)
        self.hidden_act = str(
            getattr(config, "hidden_activation", getattr(config, "hidden_act", "silu"))
        ).lower()
        self.gate_up_proj = LiteLinear(
            self.hidden_dim,
            self.num_experts * (2 * self.intermediate_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.experts.gate_up_proj",
        )
        self.down_proj = LiteLinear(
            self.intermediate_dim,
            self.num_experts * self.hidden_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.experts.down_proj",
        )
        self._cached_device: Optional[torch.device] = None
        self._cached_dtype: Optional[torch.dtype] = None
        self._cached_w1: Optional[torch.Tensor] = None
        self._cached_w2: Optional[torch.Tensor] = None
        self._expert_weight_cache: "OrderedDict[int, tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()
        self._expert_cache_device: Optional[torch.device] = None
        self._expert_cache_dtype: Optional[torch.dtype] = None
        self._max_expert_cache = max(
            0, int(os.environ.get("FASTINFERENCE_GEMMA4_MOE_EXPERT_CACHE_SIZE", "8"))
        )

    def _apply_gate_activation(self, gate: torch.Tensor) -> torch.Tensor:
        if self.hidden_act in ("gelu", "gelu_pytorch_tanh"):
            return F.gelu(gate, approximate="tanh")
        return F.silu(gate)

    def _has_awq_packed_expert_major(self) -> bool:
        qweight_gu = getattr(self.gate_up_proj, "qweight", None)
        scales_gu = getattr(self.gate_up_proj, "scales", None)
        qweight_d = getattr(self.down_proj, "qweight", None)
        scales_d = getattr(self.down_proj, "scales", None)
        return (
            isinstance(qweight_gu, torch.Tensor)
            and isinstance(scales_gu, torch.Tensor)
            and qweight_gu.ndim == 3
            and scales_gu.ndim == 3
            and isinstance(qweight_d, torch.Tensor)
            and isinstance(scales_d, torch.Tensor)
            and qweight_d.ndim == 3
            and scales_d.ndim == 3
        )

    def _materialize_one_expert_awq(
        self,
        expert_id: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._max_expert_cache > 0
            and self._expert_cache_device == device
            and self._expert_cache_dtype == dtype
        ):
            cached = self._expert_weight_cache.get(expert_id)
            if cached is not None:
                self._expert_weight_cache.move_to_end(expert_id)
                return cached
        if (
            self._expert_cache_device != device
            or self._expert_cache_dtype != dtype
        ):
            self._expert_weight_cache.clear()
            self._expert_cache_device = device
            self._expert_cache_dtype = dtype

        qweight_gu = self.gate_up_proj.qweight
        scales_gu = self.gate_up_proj.scales
        qweight_d = self.down_proj.qweight
        scales_d = self.down_proj.scales

        gsz_gu = max(1, int((qweight_gu.shape[2] * 8) // max(1, scales_gu.shape[2])))
        gsz_d = max(1, int((qweight_d.shape[2] * 8) // max(1, scales_d.shape[2])))
        w1e = dequantize_symmetric_packed_int4_pytorch(
            qweight_gu[expert_id].to(device=device, dtype=torch.int32),
            scales_gu[expert_id].to(device=device),
            group_size=gsz_gu,
        )
        w2e = dequantize_symmetric_packed_int4_pytorch(
            qweight_d[expert_id].to(device=device, dtype=torch.int32),
            scales_d[expert_id].to(device=device),
            group_size=gsz_d,
        )
        w1 = w1e[: 2 * self.intermediate_dim, : self.hidden_dim].contiguous().to(
            device=device, dtype=dtype
        )
        w2 = w2e[: self.hidden_dim, : self.intermediate_dim].contiguous().to(
            device=device, dtype=dtype
        )
        if self._max_expert_cache > 0:
            self._expert_weight_cache[expert_id] = (w1, w2)
            self._expert_weight_cache.move_to_end(expert_id)
            while len(self._expert_weight_cache) > self._max_expert_cache:
                self._expert_weight_cache.popitem(last=False)
        return w1, w2

    def _forward_awq_streaming(
        self,
        hidden_states_2d: torch.Tensor,
        router_logits: Optional[torch.Tensor],
        topk_weights: Optional[torch.Tensor] = None,
        topk_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if topk_weights is None or topk_ids is None:
            if router_logits is None:
                raise RuntimeError("router_logits or top-k routing inputs are required.")
            topk_weights, topk_ids = torch.topk(
                router_logits,
                k=self.top_k,
                dim=-1,
            )
            topk_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32).to(
                hidden_states_2d.dtype
            )
        compute_dtype = torch.float32
        out = torch.zeros_like(hidden_states_2d, dtype=compute_dtype)

        unique_experts = torch.unique(topk_ids).tolist()
        for expert_id in unique_experts:
            assignment_mask = topk_ids == int(expert_id)
            if not bool(assignment_mask.any()):
                continue
            token_idx, choice_idx = torch.nonzero(assignment_mask, as_tuple=True)
            if token_idx.numel() == 0:
                continue
            x_sel = hidden_states_2d.index_select(0, token_idx).to(compute_dtype)
            coeff = topk_weights[token_idx, choice_idx].unsqueeze(-1).to(compute_dtype)
            w1e, w2e = self._materialize_one_expert_awq(
                int(expert_id),
                hidden_states_2d.device,
                compute_dtype,
            )
            gu = F.linear(x_sel, w1e)
            g, u = torch.chunk(gu, 2, dim=-1)
            h = self._apply_gate_activation(g) * u
            y = F.linear(h, w2e) * coeff
            out.index_add_(0, token_idx, y)
        return out.to(hidden_states_2d.dtype)

    def _materialize_expert_weights(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._cached_w1 is not None
            and self._cached_w2 is not None
            and self._cached_device == device
            and self._cached_dtype == dtype
        ):
            return self._cached_w1, self._cached_w2

        qweight_gu = getattr(self.gate_up_proj, "qweight", None)
        scales_gu = getattr(self.gate_up_proj, "scales", None)
        qweight_d = getattr(self.down_proj, "qweight", None)
        scales_d = getattr(self.down_proj, "scales", None)

        # Gemma4-26B-A4B checkpoint path: expert-major packed tensors
        # gate_up_proj_packed: [E, 2I, H/8], gate_up_proj_scale: [E, 2I, H/group]
        # down_proj_packed: [E, H, I/8], down_proj_scale: [E, H, I/group]
        if (
            isinstance(qweight_gu, torch.Tensor)
            and isinstance(scales_gu, torch.Tensor)
            and qweight_gu.ndim == 3
            and scales_gu.ndim == 3
            and isinstance(qweight_d, torch.Tensor)
            and isinstance(scales_d, torch.Tensor)
            and qweight_d.ndim == 3
            and scales_d.ndim == 3
        ):
            w1_parts = []
            w2_parts = []
            gsz_gu = max(1, int((qweight_gu.shape[2] * 8) // max(1, scales_gu.shape[2])))
            gsz_d = max(1, int((qweight_d.shape[2] * 8) // max(1, scales_d.shape[2])))
            for e in range(self.num_experts):
                w1e = dequantize_symmetric_packed_int4_pytorch(
                    qweight_gu[e].to(device=device, dtype=torch.int32),
                    scales_gu[e].to(device=device),
                    group_size=gsz_gu,
                )
                w2e = dequantize_symmetric_packed_int4_pytorch(
                    qweight_d[e].to(device=device, dtype=torch.int32),
                    scales_d[e].to(device=device),
                    group_size=gsz_d,
                )
                w1_parts.append(
                    w1e[: 2 * self.intermediate_dim, : self.hidden_dim].to(device=device, dtype=dtype)
                )
                w2_parts.append(
                    w2e[: self.hidden_dim, : self.intermediate_dim].to(device=device, dtype=dtype)
                )
            w1 = torch.stack(w1_parts, dim=0).contiguous()
            w2 = torch.stack(w2_parts, dim=0).contiguous()
        else:
            gate_up_dense = _materialize_litelinear_dense_weight_awqaware(
                self.gate_up_proj,
                out_features=self.num_experts * (2 * self.intermediate_dim),
                in_features=self.hidden_dim,
                device=device,
                dtype=dtype,
            )
            down_dense = _materialize_litelinear_dense_weight_awqaware(
                self.down_proj,
                out_features=self.num_experts * self.hidden_dim,
                in_features=self.intermediate_dim,
                device=device,
                dtype=dtype,
            )
            w1 = gate_up_dense.view(
                self.num_experts,
                2 * self.intermediate_dim,
                self.hidden_dim,
            ).contiguous()
            w2 = down_dense.view(
                self.num_experts,
                self.hidden_dim,
                self.intermediate_dim,
            ).contiguous()

        self._cached_device = device
        self._cached_dtype = dtype
        self._cached_w1 = w1
        self._cached_w2 = w2
        return w1, w2

    def forward(
        self,
        hidden_states_2d: torch.Tensor,
        router_logits: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        topk_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._has_awq_packed_expert_major():
            return self._forward_awq_streaming(
                hidden_states_2d,
                router_logits,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
            )
        if topk_weights is not None and topk_ids is not None:
            router_logits = None
            # Build sparse logits proxy for fused_moe path: fill selected experts
            # with log-weights and keep others very negative.
            n_tok = int(hidden_states_2d.shape[0])
            if router_logits is None:
                router_logits = torch.full(
                    (n_tok, self.num_experts),
                    -1e9,
                    device=hidden_states_2d.device,
                    dtype=hidden_states_2d.dtype,
                )
            router_logits.scatter_(1, topk_ids, topk_weights.clamp_min(1e-20).log())
        if router_logits is None:
            raise RuntimeError("router_logits is required when top-k routing is not provided.")
        w1, w2 = self._materialize_expert_weights(
            hidden_states_2d.device,
            hidden_states_2d.dtype,
        )
        return fused_moe(
            hidden_states_2d,
            w1,
            w2,
            router_logits,
            topk=self.top_k,
            renormalize=True,
        )


class Gemma4SparseMoeBlock(nn.Module):
    def __init__(self, config: LiteConfig, quant_config: Any, prefix: str):
        super().__init__()
        self.router = Gemma4TopKRouterLite(config, quant_config, prefix)
        self.experts = Gemma4MoeExpertsLite(config, quant_config, prefix)
        self.shared_mlp = Gemma4MLP(config, quant_config, prefix)

    def forward_branches(
        self,
        hidden_states_dense: torch.Tensor,
        hidden_states_sparse: torch.Tensor,
        hidden_states_router: Optional[torch.Tensor] = None,
        lora_mapping: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states_2d, shape = _reshape_hidden_to_2d(hidden_states_sparse)
        router_src = hidden_states_router if hidden_states_router is not None else hidden_states_sparse
        router_2d, _ = _reshape_hidden_to_2d(router_src)
        router_logits, routing_weights, selected_experts = self.router(router_2d)
        sparse_out_2d = self.experts(
            hidden_states_2d,
            router_logits,
            topk_weights=routing_weights,
            topk_ids=selected_experts,
        )
        sparse_out = _restore_hidden_from_2d(sparse_out_2d, shape)
        dense_out = self.shared_mlp(hidden_states_dense, lora_mapping)
        return dense_out, sparse_out

    def forward(self, hidden_states: torch.Tensor, lora_mapping: Any = None) -> torch.Tensor:
        dense_out, sparse_out = self.forward_branches(
            hidden_states,
            hidden_states,
            hidden_states_router=hidden_states,
            lora_mapping=lora_mapping,
        )
        return dense_out + sparse_out


class Gemma4DecoderLayer(nn.Module):
    def __init__(self, config: LiteConfig, quant_config: Any, prefix: str, layer_idx: int):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=_get_eps(config))
        self.self_attn = Gemma4Attention(config, quant_config, prefix, layer_idx)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=_get_eps(config))
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=_get_eps(config))
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=_get_eps(config))
        self.pre_feedforward_layernorm_2: Optional[RMSNorm] = None
        self.post_feedforward_layernorm_1: Optional[RMSNorm] = None
        self.post_feedforward_layernorm_2: Optional[RMSNorm] = None
        self.layer_scalar = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self._fp32_residual_guard_enabled = bool(
            _is_gemma4_26b_a4b_like(config)
            and _env_truthy("FASTINFERENCE_GEMMA4_26B_FP32_RESIDUAL_GUARD")
        )
        # Backward compatible priority:
        # 1) explicit range start env
        # 2) legacy single-layer env
        # 3) default start layer 8
        guard_start_raw = os.environ.get("FASTINFERENCE_GEMMA4_26B_FP32_RESIDUAL_GUARD_START")
        if guard_start_raw is None:
            guard_start_raw = os.environ.get(
                "FASTINFERENCE_GEMMA4_26B_FP32_RESIDUAL_GUARD_LAYER", "8"
            )
        self._fp32_residual_guard_start = int(guard_start_raw)
        # Default span=3 to cover [first_drift_layer, first_drift_layer+2].
        self._fp32_residual_guard_span = max(
            1,
            int(os.environ.get("FASTINFERENCE_GEMMA4_26B_FP32_RESIDUAL_GUARD_SPAN", "3")),
        )
        self.use_moe = _is_gemma4_moe_layer(config, layer_idx)
        if self.use_moe:
            # Gemma4-26B-A4B checkpoints expose dual pre-FFN norms and dual
            # branch post-FFN norms at layer root.
            self.pre_feedforward_layernorm_2 = RMSNorm(config.hidden_size, eps=_get_eps(config))
            self.post_feedforward_layernorm_1 = RMSNorm(config.hidden_size, eps=_get_eps(config))
            self.post_feedforward_layernorm_2 = RMSNorm(config.hidden_size, eps=_get_eps(config))
            self.mlp = Gemma4SparseMoeBlock(config, quant_config, prefix)
        else:
            self.mlp = Gemma4MLP(config, quant_config, prefix)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: Any,
        attn_metadata: Any,
        lora_mapping: Any = None,
    ) -> torch.Tensor:
        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn(h, positions, kv_cache, attn_metadata, lora_mapping)
        h = self.post_attention_layernorm(h)
        guard_hit = (
            self._fp32_residual_guard_enabled
            and self._fp32_residual_guard_start
            <= self.layer_idx
            < (self._fp32_residual_guard_start + self._fp32_residual_guard_span)
        )
        if guard_hit:
            x = _residual_add_fp32(residual, h)
        else:
            x = residual + h

        residual = x
        h_dense = self.pre_feedforward_layernorm(x)
        if self.use_moe and isinstance(self.mlp, Gemma4SparseMoeBlock):
            # Match HF Gemma4 MoE flow:
            # - dense MLP branch consumes pre_feedforward_layernorm(residual)
            # - router consumes raw residual (before pre-FF norms)
            # - sparse experts consume pre_feedforward_layernorm_2(residual)
            dense_out = self.mlp.shared_mlp(h_dense, lora_mapping=lora_mapping)
            if self.post_feedforward_layernorm_1 is not None:
                dense_out = self.post_feedforward_layernorm_1(dense_out)

            router_in_2d, router_shape = _reshape_hidden_to_2d(residual)
            router_logits, routing_weights, selected_experts = self.mlp.router(
                router_in_2d
            )
            if self.pre_feedforward_layernorm_2 is not None:
                sparse_in = self.pre_feedforward_layernorm_2(residual)
            else:
                sparse_in = residual
            sparse_in_2d, _ = _reshape_hidden_to_2d(sparse_in)
            sparse_out_2d = self.mlp.experts(
                sparse_in_2d,
                router_logits,
                topk_weights=routing_weights,
                topk_ids=selected_experts,
            )
            sparse_out = _restore_hidden_from_2d(sparse_out_2d, router_shape)
            if self.post_feedforward_layernorm_2 is not None:
                sparse_out = self.post_feedforward_layernorm_2(sparse_out)
            h = dense_out + sparse_out
        else:
            h = self.mlp(h_dense, lora_mapping)
        h = self.post_feedforward_layernorm(h)
        if guard_hit:
            x = _residual_add_fp32(residual, h)
        else:
            x = residual + h
        return x * self.layer_scalar


class Gemma4TextModel(nn.Module):
    def __init__(self, hf_config: Any, quant_config: Any, prefix: str = "model"):
        super().__init__()
        self.config = LiteConfig(hf_config)
        padding_idx = int(getattr(hf_config, "pad_token_id", 0) or 0)
        self.embed_scale = float(self.config.hidden_size) ** 0.5
        self.embed_tokens = nn.Embedding(
            self.config.vocab_size,
            self.config.hidden_size,
            padding_idx=padding_idx,
        )
        self.layers = nn.ModuleList(
            [
                Gemma4DecoderLayer(
                    self.config, quant_config, prefix=f"layers.{i}", layer_idx=i
                )
                for i in range(self.config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(self.config.hidden_size, eps=_get_eps(self.config))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Any,
        attn_metadata: Any,
        lora_mapping: Any = None,
    ) -> torch.Tensor:
        if input_ids.dtype == torch.long:
            x = self.embed_tokens(input_ids) * self.embed_scale
        else:
            x = input_ids
        for i, layer in enumerate(self.layers):
            x = layer(x, positions, kv_caches[i], attn_metadata, lora_mapping)
        return self.norm(x)


def _assert_text_only_kwargs(kwargs: dict[str, Any]) -> None:
    banned = (
        "pixel_values",
        "image_embeds",
        "audio_values",
        "input_features",
        "multimodal_embeddings",
    )
    for k in banned:
        if k in kwargs and kwargs[k] is not None:
            raise NotImplementedError(
                f"Gemma4 text-only path does not support multimodal input: {k}"
            )


class Gemma4ForConditionalGeneration(nn.Module):
    def __init__(self, vllm_config: Any, prefix: str = ""):
        super().__init__()
        hf_config = vllm_config.model_config.hf_config
        self.model = Gemma4TextModel(hf_config, vllm_config.quant_config, prefix="model")
        self.lm_head = LiteLinear(
            self.model.config.hidden_size,
            self.model.config.vocab_size,
            bias=False,
            prefix="lm_head",
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Any,
        attn_metadata: Any,
        lora_mapping: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        _assert_text_only_kwargs(kwargs)
        hidden = self.model(input_ids, positions, kv_caches, attn_metadata, lora_mapping)
        if getattr(self.model.config, "tie_word_embeddings", False):
            logits = torch.nn.functional.linear(hidden[:, -1:, :], self.model.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden[:, -1:, :], lora_mapping)
        final_softcap = getattr(self.model.config, "final_logit_softcapping", None)
        if final_softcap is not None and float(final_softcap) > 0:
            logits = torch.tanh(logits / float(final_softcap)) * float(final_softcap)
        return logits


class Gemma4ForCausalLM(Gemma4ForConditionalGeneration):
    pass
