# SPDX-License-Identifier: Apache-2.0
# GatedDeltaNet helpers below align with Hugging Face `modeling_qwen3_5.py` (torch fallback path).
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Any, Tuple
from vllm.model_executor.layers.lite_linear import LiteLinear
from .lite_config import LiteConfig


def _env_truthy(name: str) -> bool:
    """True for 1/true/yes/on (case-insensitive). Used for debug / ablation toggles."""
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def _torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class Qwen3_5RMSNormGated(nn.Module):
    """Matches HF `Qwen3_5RMSNormGated` (non-fused)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: Optional[torch.Tensor] = None):
        if gate is None:
            raise ValueError("Qwen3_5RMSNormGated requires gate")
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


class Qwen3_5RMSNorm(nn.Module):
    """
    Matches HF `Qwen3_5RMSNorm`: scale is (1 + weight), not Llama-style `weight` alone.
    See transformers Qwen3.5 modeling (PR #29402 note in HF source).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


def _fp16_safe_abs_max() -> float:
    return float(torch.finfo(torch.float16).max) * 0.92


def _scale_tensor_to_abs_cap(t: torch.Tensor, cap: float) -> torch.Tensor:
    """Uniform scale-down if |t| exceeds cap (preserves direction vs per-element clamp)."""
    m = t.detach().abs().max()
    if not torch.isfinite(m) or m <= cap:
        return t
    return t * (cap / m.clamp(min=1e-6))


def _cap_residual_delta_rms(x_f: torch.Tensor, delta_f: torch.Tensor, factor: float) -> torch.Tensor:
    """
    Scale down delta when its per-token RMS far exceeds the stream RMS.
    Reduces saturation from the simplified linear-attn path while keeping small deltas intact.
    """
    rx = x_f.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
    rd = delta_f.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
    cap = factor * rx
    s = torch.minimum(torch.ones_like(rd), cap / rd)
    return delta_f * s


def _residual_merge_fp16(x_f: torch.Tensor, delta_f: torch.Tensor, out_dtype: torch.dtype, factor: float) -> torch.Tensor:
    # FASTINFERENCE_DISABLE_RESIDUAL_STABILIZER=1: skip RMS cap and output abs-cap (numerical ablation vs HF).
    if _env_truthy("FASTINFERENCE_DISABLE_RESIDUAL_STABILIZER"):
        y = x_f + delta_f
        y = torch.nan_to_num(
            y, nan=0.0, posinf=_fp16_safe_abs_max(), neginf=-_fp16_safe_abs_max()
        )
        return y.to(out_dtype)
    d = _cap_residual_delta_rms(x_f, delta_f, factor)
    y = x_f + d
    y = torch.nan_to_num(y, nan=0.0, posinf=_fp16_safe_abs_max(), neginf=-_fp16_safe_abs_max())
    y = _scale_tensor_to_abs_cap(y, _fp16_safe_abs_max())
    return y.to(out_dtype)


def _call_litelinear_stable(layer: LiteLinear, x_f32: torch.Tensor, pre_cap: float = 2048.0) -> torch.Tensor:
    """
    Call LiteLinear through its module path (for hook visibility) while limiting fp16 overflow risk.
    We scale only the input magnitude before cast; no inverse-rescale is applied.
    Set FASTINFERENCE_DISABLE_LINEAR_INPUT_CAP=1 to skip input scaling (ablation / alignment with reference).
    """
    if layer.weight.numel() == 0:
        out_shape = (*x_f32.shape[:-1], layer.output_size)
        return torch.zeros(out_shape, device=x_f32.device, dtype=torch.float32)
    if _env_truthy("FASTINFERENCE_DISABLE_LINEAR_INPUT_CAP"):
        x_safe = x_f32
    else:
        x_safe = _scale_tensor_to_abs_cap(x_f32, pre_cap)
    out = layer(x_safe.to(layer.weight.dtype))
    return out.float()


class Qwen3_5LinearAttentionLayer(nn.Module):
    """
    GatedDeltaNet linear attention aligned with HF `Qwen3_5GatedDeltaNet` (torch fallback).
    When `attn_metadata` contains `linear_conv_carry` + `linear_attn_carry` (see LiteEngine),
    applies causal conv streaming + recurrent delta-net state across decode / chunked prefill.
    """

    def __init__(self, config: LiteConfig, quant_config, prefix="", layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim

        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))
        self.linear_attn = nn.Module()
        self.linear_attn.in_proj_qkv = LiteLinear(
            config.hidden_size,
            self.conv_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_attn.in_proj_qkv",
        )
        self.linear_attn.in_proj_z = LiteLinear(
            config.hidden_size,
            self.value_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_attn.in_proj_z",
        )
        self.linear_attn.out_proj = LiteLinear(
            self.value_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_attn.out_proj",
        )
        self.linear_attn.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            bias=False,
        )
        # HF default init; checkpoint overwrites.
        _a_init = torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(0, 16)
        self.linear_attn.A_log = nn.Parameter(torch.log(_a_init))
        self.linear_attn.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.linear_attn.in_proj_a = LiteLinear(
            config.hidden_size,
            self.num_v_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_attn.in_proj_a",
        )
        self.linear_attn.in_proj_b = LiteLinear(
            config.hidden_size,
            self.num_v_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_attn.in_proj_b",
        )
        self.linear_attn.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)

        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))
        self.mlp = nn.Module()
        self.mlp.gate_proj = LiteLinear(
            config.hidden_size, config.intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.gate_proj"
        )
        self.mlp.up_proj = LiteLinear(
            config.hidden_size, config.intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.up_proj"
        )
        self.mlp.down_proj = LiteLinear(
            config.intermediate_size, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.down_proj"
        )

    def forward(self, x, positions, kv_cache, attn_metadata):
        input_dtype = x.dtype
        residual = x
        h = self.input_layernorm(x)
        batch_size, seq_len, _ = h.shape

        mixed = self.linear_attn.in_proj_qkv(h)
        k = self.conv_kernel_size
        use_stream = (
            attn_metadata is not None
            and attn_metadata.get("linear_attn_carry") is not None
            and attn_metadata.get("linear_conv_carry") is not None
        )

        if use_stream:
            conv_carry_list = attn_metadata["linear_conv_carry"]
            attn_carry_list = attn_metadata["linear_attn_carry"]
            w = self.linear_attn.conv1d.weight
            conv_dtype = w.dtype
            # (B, C, L) — conv input (pre-SiLU), same as HF in_proj_qkv layout transposed
            x_legacy = mixed.transpose(1, 2).contiguous().to(dtype=conv_dtype)
            prev_cc = (
                conv_carry_list[self.layer_idx]
                if self.layer_idx < len(conv_carry_list)
                else None
            )
            # First prefill chunk with seq_len>1: use checkpoint-matched nn.Conv1d (symmetric pad + trim).
            # Streaming F.conv1d+left-pad differs slightly and can collapse logits; decode / chunked tail uses cat path.
            use_checkpoint_conv = prev_cc is None and seq_len > 1
            if use_checkpoint_conv:
                mixed_conv = F.silu(
                    self.linear_attn.conv1d(x_legacy)[:, :, :seq_len]
                )
                mixed_qkv = mixed_conv.transpose(1, 2)
                x_for_carry = F.pad(x_legacy, (k - 1, 0))
            else:
                x_c = x_legacy
                if prev_cc is not None:
                    x_c = torch.cat([prev_cc, x_c], dim=-1)
                else:
                    x_c = F.pad(x_c, (k - 1, 0))
                bias = self.linear_attn.conv1d.bias
                mixed_conv = F.conv1d(
                    x_c,
                    w,
                    bias,
                    stride=1,
                    padding=0,
                    groups=self.conv_dim,
                )
                mixed_conv = F.silu(mixed_conv)
                if mixed_conv.shape[-1] != seq_len:
                    raise RuntimeError(
                        f"linear_attn conv length mismatch: got {mixed_conv.shape[-1]}, expected {seq_len} "
                        f"(layer={self.layer_idx}, k={k})"
                    )
                mixed_qkv = mixed_conv.transpose(1, 2)
                x_for_carry = x_c
            if self.layer_idx < len(conv_carry_list):
                conv_carry_list[self.layer_idx] = (
                    x_for_carry[:, :, -k + 1 :].contiguous().detach()
                )
        else:
            mixed_qkv = mixed.transpose(1, 2)
            mixed_qkv = F.silu(self.linear_attn.conv1d(mixed_qkv)[:, :, :seq_len])
            mixed_qkv = mixed_qkv.transpose(1, 2)

        z = self.linear_attn.in_proj_z(h)
        z = z.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        b = self.linear_attn.in_proj_b(h)
        a = self.linear_attn.in_proj_a(h)

        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )

        query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.linear_attn.A_log.float().exp() * F.softplus(a.float() + self.linear_attn.dt_bias.float())

        head_expand = self.num_v_heads // self.num_k_heads
        if head_expand > 1:
            query = query.repeat_interleave(head_expand, dim=2)
            key = key.repeat_interleave(head_expand, dim=2)

        incoming_state = None
        attn_carry_list = (
            attn_metadata["linear_attn_carry"] if use_stream else None
        )
        if use_stream and attn_carry_list is not None and self.layer_idx < len(
            attn_carry_list
        ):
            incoming_state = attn_carry_list[self.layer_idx]

        core_attn_out, last_state = _torch_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=incoming_state,
            output_final_state=use_stream,
            use_qk_l2norm_in_kernel=True,
        )
        if (
            use_stream
            and attn_carry_list is not None
            and self.layer_idx < len(attn_carry_list)
        ):
            attn_carry_list[self.layer_idx] = (
                None if last_state is None else last_state.detach()
            )

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.linear_attn.norm(core_attn_out, z_flat)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, self.value_dim)

        attn_delta = self.linear_attn.out_proj(core_attn_out.to(input_dtype))
        hidden_states = residual + attn_delta

        h_post = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp.down_proj(F.silu(self.mlp.gate_proj(h_post)) * self.mlp.up_proj(h_post))
        return hidden_states + mlp_out

class Qwen3_5FullAttentionLayer(nn.Module):
    """
    Full attention matching HF `Qwen3_5Attention`: fused q/gate from q_proj, per-head q_norm/k_norm,
    sigmoid gate on attention output before o_proj.
    """

    def __init__(self, config: LiteConfig, quant_config, prefix=""):
        super().__init__()
        self.config = config
        eps = getattr(config, "rms_norm_eps", 1e-6)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        q_out = self.num_heads * self.head_dim * 2
        kv_out = self.num_kv_heads * self.head_dim
        attn_out_dim = self.num_heads * self.head_dim

        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=eps)
        self.self_attn = nn.Module()
        self.self_attn.q_proj = LiteLinear(
            config.hidden_size, q_out, bias=True, quant_config=quant_config, prefix=f"{prefix}.self_attn.q_proj"
        )
        self.self_attn.k_proj = LiteLinear(
            config.hidden_size, kv_out, bias=True, quant_config=quant_config, prefix=f"{prefix}.self_attn.k_proj"
        )
        self.self_attn.v_proj = LiteLinear(
            config.hidden_size, kv_out, bias=True, quant_config=quant_config, prefix=f"{prefix}.self_attn.v_proj"
        )
        self.self_attn.o_proj = LiteLinear(
            attn_out_dim, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.o_proj"
        )
        self.self_attn.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=eps)
        self.self_attn.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=eps)
        self.mlp = nn.Module()
        self.mlp.gate_proj = LiteLinear(
            config.hidden_size, config.intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.gate_proj"
        )
        self.mlp.up_proj = LiteLinear(
            config.hidden_size, config.intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.up_proj"
        )
        self.mlp.down_proj = LiteLinear(
            config.intermediate_size, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.down_proj"
        )
        from .llama import get_rotary_embedding

        self.rotary_emb = get_rotary_embedding(config)
        # Legacy ROCm-friendly caps on full-attn matmuls / residuals (off by default for HF parity).
        self._use_full_attn_stabilizer = _env_truthy("FASTINFERENCE_QWEN35_FULLATTN_STABILIZER")

    def forward(self, x, positions, kv_cache, attn_metadata):
        input_dtype = x.dtype
        h = self.input_layernorm(x)
        bs, seq, _ = h.shape
        nh, nkv, hd = self.num_heads, self.num_kv_heads, self.head_dim
        n_tokens = bs * seq

        q_merged = self.self_attn.q_proj(h)
        q_part, gate = torch.chunk(q_merged.view(bs, seq, nh, 2 * hd), 2, dim=-1)
        gate = gate.reshape(bs, seq, nh * hd)
        q_part = self.self_attn.q_norm(q_part)
        k = self.self_attn.k_norm(self.self_attn.k_proj(h).view(bs, seq, nkv, hd))
        v = self.self_attn.v_proj(h).view(bs, seq, nkv, hd)

        q = q_part.reshape(n_tokens, nh, hd)
        k = k.reshape(n_tokens, nkv, hd)
        v = v.reshape(n_tokens, nkv, hd)
        q, k = self.rotary_emb(positions, q.unsqueeze(0), k.unsqueeze(0))
        q = q.squeeze(0)
        k = k.squeeze(0)
        k_cache, v_cache = kv_cache
        from vllm.kernels.triton.reshape_and_cache import reshape_and_cache
        from vllm.kernels.triton.paged_attention import paged_attention_v1

        reshape_and_cache(k, v, k_cache, v_cache, attn_metadata["slot_mapping"], "auto")
        attn_in = torch.empty((n_tokens, nh, hd), device=q.device, dtype=q.dtype)
        block_tables = attn_metadata["block_tables"]
        seq_lens = attn_metadata["seq_lens"]
        is_prefill = attn_metadata.get("is_prefill", False)
        max_ctx = max(nh * hd, getattr(self.config, "max_position_embeddings", 4096))

        if seq > 1 and is_prefill:
            end_pos = int(seq_lens[0].item())
            start_pos = end_pos - seq
            seq_lens_ext = torch.arange(start_pos + 1, end_pos + 1, device=q.device, dtype=torch.int32)
            block_tables_ext = block_tables.expand(seq, -1).contiguous()
            paged_attention_v1(
                attn_in,
                q.contiguous(),
                k_cache,
                v_cache,
                nh,
                hd**-0.5,
                block_tables_ext,
                seq_lens_ext,
                k_cache.shape[1],
                max_ctx,
                None,
                "auto",
                None,
                None,
                num_kv_heads=nkv,
            )
        else:
            paged_attention_v1(
                attn_in,
                q.contiguous(),
                k_cache,
                v_cache,
                nh,
                hd**-0.5,
                block_tables,
                seq_lens,
                k_cache.shape[1],
                max_ctx,
                None,
                "auto",
                None,
                None,
                num_kv_heads=nkv,
            )

        attn_flat = attn_in.view(bs, seq, nh * hd)
        attn_gated = attn_flat * torch.sigmoid(gate.to(dtype=attn_flat.dtype))

        if self._use_full_attn_stabilizer:
            attn_out = _call_litelinear_stable(self.self_attn.o_proj, attn_gated.float(), pre_cap=1024.0)
            hidden_states = _residual_merge_fp16(x.float(), attn_out, input_dtype, 8.0)
        else:
            attn_out = self.self_attn.o_proj(attn_gated)
            hidden_states = x + attn_out

        h_post = self.post_attention_layernorm(hidden_states)
        if self._use_full_attn_stabilizer:
            gp = self.mlp.gate_proj(h_post).float()
            up = self.mlp.up_proj(h_post).float()
            mlp_mid = F.silu(gp) * up
            mlp_out = _call_litelinear_stable(self.mlp.down_proj, mlp_mid, pre_cap=1536.0)
            return _residual_merge_fp16(hidden_states.float(), mlp_out, input_dtype, 8.0)

        mlp_out = self.mlp.down_proj(F.silu(self.mlp.gate_proj(h_post)) * self.mlp.up_proj(h_post))
        return hidden_states + mlp_out

class Qwen2Model(nn.Module):
    def __init__(self, hf_config, quant_config):
        super().__init__()
        self.config = LiteConfig(hf_config)
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.layers = nn.ModuleList()
        for i in range(self.config.num_hidden_layers):
            if (i % 4) != 3:
                self.layers.append(
                    Qwen3_5LinearAttentionLayer(
                        self.config, quant_config, f"model.layers.{i}", layer_idx=i
                    )
                )
            else:
                self.layers.append(
                    Qwen3_5FullAttentionLayer(
                        self.config, quant_config, f"model.layers.{i}"
                    )
                )
        self.norm = Qwen3_5RMSNorm(self.config.hidden_size, eps=getattr(self.config, "rms_norm_eps", 1e-6))

class Qwen3_5ForConditionalGeneration(nn.Module):
    def __init__(self, vllm_config, prefix=""):
        super().__init__()
        self.model = Qwen2Model(vllm_config.model_config.hf_config, vllm_config.quant_config)
        self.lm_head = nn.Linear(self.model.config.hidden_size, self.model.config.vocab_size, bias=False)
    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        x = self.model.embed_tokens(input_ids)
        for i in range(len(self.model.layers)): x = self.model.layers[i](x, positions, kv_caches[i], attn_metadata)
        return self.lm_head(self.model.norm(x))

class Qwen3_5MoeForConditionalGeneration(Qwen3_5ForConditionalGeneration): pass
