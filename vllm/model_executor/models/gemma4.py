# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear
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
    s = q.shape[2]
    causal = torch.triu(torch.ones(s, s, device=q.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal, float("-inf"))
    if local_window is not None and local_window > 0:
        idx = torch.arange(s, device=q.device)
        dist = idx[None, :] - idx[:, None]
        local_mask = dist >= local_window
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


def _gather_recent_kv(
    kv_cache: Any,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    batch_idx: int,
    num_kv_heads: int,
    head_dim: int,
    local_window: int,
    kv_cache_dtype: str,
    kv_scale_cache: Optional[tuple[Any, Any]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    k_cache, v_cache = kv_cache
    block_size = int(k_cache.shape[1])
    seq_len = int(seq_lens[batch_idx].item())
    start = max(0, seq_len - local_window)
    k_rows = []
    v_rows = []
    k_scale_cache = kv_scale_cache[0] if kv_scale_cache is not None else None
    v_scale_cache = kv_scale_cache[1] if kv_scale_cache is not None else None
    is_int4 = "int4" in str(kv_cache_dtype).lower()

    for token_idx in range(start, seq_len):
        block_idx = int(block_tables[batch_idx, token_idx // block_size].item())
        block_offset = token_idx % block_size
        if is_int4:
            k_row = _decode_int4_row(
                k_cache, k_scale_cache, block_idx, block_offset, num_kv_heads, head_dim
            )
            v_row = _decode_int4_row(
                v_cache, v_scale_cache, block_idx, block_offset, num_kv_heads, head_dim
            )
        else:
            k_row = k_cache[block_idx, block_offset, :num_kv_heads, :head_dim].to(torch.float32)
            v_row = v_cache[block_idx, block_offset, :num_kv_heads, :head_dim].to(torch.float32)
        k_rows.append(k_row)
        v_rows.append(v_row)

    k = torch.stack(k_rows, dim=0).unsqueeze(0)
    v = torch.stack(v_rows, dim=0).unsqueeze(0)
    return k, v


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
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    self.scale,
                    local_window=local_window,
                    softcap=softcap,
                ).view(bsz, seqlen, -1)
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
                    k_ctx, v_ctx = _gather_recent_kv(
                        kv_cache=kv_cache,
                        block_tables=block_tables,
                        seq_lens=seq_lens,
                        batch_idx=bi,
                        num_kv_heads=self.num_kv_heads,
                        head_dim=self.head_dim,
                        local_window=int(local_window or 0),
                        kv_cache_dtype=str(kv_dtype_name),
                        kv_scale_cache=(k_scale_cache, v_scale_cache),
                    )
                    q_i = q[bi : bi + 1].transpose(1, 2)
                    out_i = _causal_attention_ref(
                        q_i,
                        k_ctx.transpose(1, 2).to(q.dtype),
                        v_ctx.transpose(1, 2).to(q.dtype),
                        self.scale,
                        local_window=None,
                        softcap=softcap,
                    ).view(1, seqlen, -1)
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


class Gemma4DecoderLayer(nn.Module):
    def __init__(self, config: LiteConfig, quant_config: Any, prefix: str, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=_get_eps(config))
        self.self_attn = Gemma4Attention(config, quant_config, prefix, layer_idx)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=_get_eps(config))
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=_get_eps(config))
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=_get_eps(config))
        self.layer_scalar = nn.Parameter(torch.tensor(1.0), requires_grad=False)
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
        x = residual + h

        residual = x
        h = self.pre_feedforward_layernorm(x)
        h = self.mlp(h, lora_mapping)
        h = self.post_feedforward_layernorm(h)
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
