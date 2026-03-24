# SPDX-License-Identifier: Apache-2.0
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Any, Tuple
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear
from .lite_config import LiteConfig
from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2RotaryEmbedding,
    apply_rotary_emb,
)

USE_TRITON = False # Stable PyTorch Path for AMD Audit


def _deepseek_routed_experts_hf_style(
    hidden_states: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    act_fn,
) -> torch.Tensor:
    """
    Match ``transformers`` ``DeepseekV2Experts.forward`` (per-expert loop + index_add).
    ``gate_up`` is ``[num_experts, 2 * intermediate, hidden]`` (concat of gate_proj and up_proj).
    ``down`` is ``[num_experts, hidden, intermediate]``.
    ``topk_ids`` / ``topk_weights`` are ``[num_tokens, top_k]`` (post-softmax routing).
    """
    num_experts = int(gate_up.shape[0])
    final_hidden_states = torch.zeros_like(hidden_states)
    expert_mask = torch.nn.functional.one_hot(topk_ids, num_classes=num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate, up = torch.nn.functional.linear(current_state, gate_up[expert_idx]).chunk(2, dim=-1)
        current_hidden_states = act_fn(gate) * up
        current_hidden_states = torch.nn.functional.linear(current_hidden_states, down[expert_idx])
        current_hidden_states = current_hidden_states * topk_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
    return final_hidden_states


def _deepseek_attn_fp32_qk() -> bool:
    """Compute QK^T attention scores in float32 (closer to stable eager matmul + softmax). Enable with FASTINFERENCE_DEEPSEEK_ATTN_FP32=1."""
    v = os.environ.get("FASTINFERENCE_DEEPSEEK_ATTN_FP32", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _qk_attn_scores(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    if _deepseek_attn_fp32_qk():
        return torch.matmul(query_states.float(), key_states.float().transpose(-2, -1)) * scale
    return torch.matmul(query_states, key_states.transpose(-2, -1)) * scale


def _sync_yarn_rope_from_scaling(rp: dict, rs: dict | None) -> None:
    """
    HF DeepSeek-V2 Chat configs often put YaRN under rope_scaling.type=yarn but omit
    rope_parameters.rope_type and factor. DeepseekV2RotaryEmbedding reads rope_type/factor
    from rope_parameters; without them, RoPE falls back to default and GGUF logits drift.
    """
    if not isinstance(rs, dict) or rs.get("type") != "yarn":
        return
    rp["rope_type"] = "yarn"
    if "factor" not in rs:
        return
    if "factor" not in rp:
        rp["factor"] = float(rs["factor"])


def normalize_deepseek_rope_config_dict(data: dict) -> dict:
    """
    Normalize rope_parameters / rope_scaling in a raw config.json dict (tokenizer load path).
    Mutates `data` in place and returns it.
    """
    if data.get("model_type") != "deepseek_v2":
        return data
    rs = data.get("rope_scaling")
    rp = data.get("rope_parameters")
    if not isinstance(rp, dict):
        rp = {}
        data["rope_parameters"] = rp
    if isinstance(rs, dict):
        for k in ("mscale", "mscale_all_dim", "beta_fast", "beta_slow"):
            if k in rs and k not in rp:
                rp[k] = rs[k]
        if "original_max_position_embeddings" in rs and "original_max_position_embeddings" not in rp:
            rp["original_max_position_embeddings"] = rs["original_max_position_embeddings"]
    if "rope_theta" not in rp and data.get("rope_theta") is not None:
        rp["rope_theta"] = float(data["rope_theta"])
    for container in (rp, rs if isinstance(rs, dict) else None):
        if container is None:
            continue
        for k in ("factor", "beta_fast", "beta_slow", "mscale", "mscale_all_dim", "rope_theta"):
            if k in container and isinstance(container[k], int):
                container[k] = float(container[k])
    _sync_yarn_rope_from_scaling(rp, rs if isinstance(rs, dict) else None)
    return data


def patch_deepseek_config_json_for_tokenizer(model_path: str) -> None:
    """
    Idempotently rewrite local config.json rope fields so AutoTokenizer / HF
    validation does not warn on int factors (e.g. 40 vs 40.0).
    """
    import json
    import os

    cfg_path = os.path.join(str(model_path), "config.json")
    if not os.path.isfile(cfg_path):
        return
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return
    if data.get("model_type") != "deepseek_v2":
        return
    before = json.dumps(data, sort_keys=True)
    normalize_deepseek_rope_config_dict(data)
    after = json.dumps(data, sort_keys=True)
    if before == after:
        return
    try:
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def _normalize_deepseek_rope_config(config: Any) -> None:
    """
    HF transformers validates rope_parameters numeric fields as float (not int).
    YaRN also expects mscale / beta_* in rope_parameters — merge from rope_scaling when present.
    """
    rs = getattr(config, "rope_scaling", None)
    rp = getattr(config, "rope_parameters", None)
    if not isinstance(rp, dict):
        rp = {}
        setattr(config, "rope_parameters", rp)
    if isinstance(rs, dict):
        for k in ("mscale", "mscale_all_dim", "beta_fast", "beta_slow"):
            if k in rs and k not in rp:
                rp[k] = rs[k]
        if "original_max_position_embeddings" in rs and "original_max_position_embeddings" not in rp:
            rp["original_max_position_embeddings"] = rs["original_max_position_embeddings"]
    rt = getattr(config, "rope_theta", None)
    if rt is not None and "rope_theta" not in rp:
        rp["rope_theta"] = float(rt)
    for container in (rp, rs if isinstance(rs, dict) else None):
        if container is None:
            continue
        for k in ("factor", "beta_fast", "beta_slow", "mscale", "mscale_all_dim", "rope_theta"):
            if k in container and isinstance(container[k], int):
                container[k] = float(container[k])
    _sync_yarn_rope_from_scaling(rp, rs if isinstance(rs, dict) else None)


class DeepseekV2MoE(nn.Module):
    def __init__(self, config: LiteConfig, quant_config, prefix: str):
        super().__init__()
        self.config = config
        self.num_experts = getattr(config, "n_routed_experts", 64)
        self.top_k = getattr(config, "num_experts_per_tok", 6)
        self.router = LiteLinear(config.hidden_size, self.num_experts, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.gate_inp")
        self.shared_experts = nn.Module()
        # HF DeepSeek MoE: shared experts use moe_intermediate_size per expert; GGUF fuses n_shared_experts into one FFN width.
        n_shexp = int(getattr(config, "n_shared_experts", 1) or 1)
        moe_ish = int(getattr(config, "moe_intermediate_size", 1408) or 1408)
        sh_inter = n_shexp * moe_ish
        self.shared_experts.gate_proj = LiteLinear(config.hidden_size, sh_inter, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.gate_shexp")
        self.shared_experts.up_proj = LiteLinear(config.hidden_size, sh_inter, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.up_shexp")
        self.shared_experts.down_proj = LiteLinear(sh_inter, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.down_shexp")
        self.experts_gate = nn.Parameter(torch.empty(0), requires_grad=False)
        self.experts_up = nn.Parameter(torch.empty(0), requires_grad=False)
        self.experts_down = nn.Parameter(torch.empty(0), requires_grad=False)
        self.routed_scaling_factor = float(getattr(config, "routed_scaling_factor", 1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, seq, h = x.shape
        x_flat = x.view(-1, h)
        shared_out = self.shared_experts.down_proj(
            F.silu(self.shared_experts.gate_proj(x)) * self.shared_experts.up_proj(x)
        )

        # If routed experts are not present in this checkpoint, fall back to shared experts only.
        if (
            self.experts_gate.numel() == 0
            or self.experts_up.numel() == 0
            or self.experts_down.numel() == 0
        ):
            return shared_out

        # Match HF DeepseekV2Moe.route_tokens_to_experts: softmax + topk on float32 (no bf16 cast before topk).
        rw = self.router.weight
        router_logits = F.linear(x_flat.float(), rw.float())
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(routing_weights, self.top_k, dim=-1, sorted=False)
        topk_weights = topk_weights * self.routed_scaling_factor

        gate_up = torch.cat([self.experts_gate, self.experts_up], dim=1)
        ha = getattr(self.config, "hidden_act", "silu") or "silu"
        act_fn = F.silu if ha == "silu" else F.gelu
        routed_out = _deepseek_routed_experts_hf_style(
            x_flat,
            gate_up,
            self.experts_down,
            topk_ids,
            topk_weights,
            act_fn,
        )
        return shared_out + routed_out.view(bs, seq, h)

class DeepseekV2DecoderLayer(nn.Module):
    def __init__(self, config: LiteConfig, quant_config, prefix=""):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.qk_nope_dim = getattr(config, "qk_nope_head_dim", 64)
        self.qk_rope_dim = getattr(config, "qk_rope_head_dim", 64)
        self.v_head_dim = getattr(config, "v_head_dim", 128)
        raw_q_lora_rank = getattr(config, "q_lora_rank", None)
        self.q_lora_rank = raw_q_lora_rank if raw_q_lora_rank is not None else 0
        self.use_q_lora = self.q_lora_rank > 0
        self.kv_lora_rank = getattr(config, "kv_lora_rank", 512)
        _rms_eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.input_layernorm = RMSNorm(config.hidden_size, eps=_rms_eps)
        if self.use_q_lora:
            self.q_a_proj = LiteLinear(config.hidden_size, self.q_lora_rank, bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.q_a_proj")
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=_rms_eps)
            self.q_b_proj = LiteLinear(self.q_lora_rank, self.num_heads * (self.qk_nope_dim + self.qk_rope_dim), bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.q_b_proj")
        else:
            # Some DeepSeek V2 Lite checkpoints set q_lora_rank to null and use a direct q projection.
            self.q_proj = LiteLinear(
                config.hidden_size,
                self.num_heads * (self.qk_nope_dim + self.qk_rope_dim),
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn.q_proj",
            )
            self.q_a_proj = None
            self.q_a_layernorm = None
            self.q_b_proj = None
        self.kv_a_proj = LiteLinear(config.hidden_size, self.kv_lora_rank + self.qk_rope_dim, bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.kv_a_proj")
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=_rms_eps)
        self.kv_b_proj = LiteLinear(self.kv_lora_rank, self.num_heads * (self.qk_nope_dim + self.v_head_dim), bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.kv_b_proj")
        self.o_proj = LiteLinear(self.num_heads * self.v_head_dim, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.o_proj")
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=_rms_eps)
        # RoPE lives on DeepseekV2Model (single module) so YaRN/dynamic_rope_update matches HF — not per-layer.
        self.q_head_dim = self.qk_nope_dim + self.qk_rope_dim
        self.softmax_scale = self.q_head_dim ** (-0.5)
        is_moe = int(prefix.split(".")[-1]) >= getattr(config, "first_k_dense_replace", 1)
        if is_moe: self.mlp = DeepseekV2MoE(config, quant_config, prefix)
        else:
            self.mlp = nn.Module()
            self.mlp.gate_proj = LiteLinear(config.hidden_size, config.intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.gate_proj")
            self.mlp.up_proj = LiteLinear(config.hidden_size, config.intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.up_proj")
            self.mlp.down_proj = LiteLinear(config.intermediate_size, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.down_proj")
            self.mlp.forward = lambda x: self.mlp.down_proj(F.silu(self.mlp.gate_proj(x)) * self.mlp.up_proj(x))

    def forward(self, x, positions, kv_cache, attn_metadata, lora_mapping=None, freqs_cis=None):
        h = self.input_layernorm(x)
        bs, seq, _ = h.shape
        position_ids = positions.long()
        md = attn_metadata if isinstance(attn_metadata, dict) else {}
        mla_kv = md.get("mla_kv")
        mla_layer_idx = int(md.get("mla_layer_idx", 0))
        mla_cached_len = int(md.get("mla_cached_len", 0))
        is_prefill = bool(md.get("is_prefill", True))
        use_mla_cache = mla_kv is not None and mla_layer_idx < len(mla_kv)

        if self.use_q_lora:
            q_full = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(h)))
        else:
            q_full = self.q_proj(h)
        q = q_full.view(bs, seq, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_dim, self.qk_rope_dim], dim=-1)

        compressed_kv = self.kv_a_proj(h)
        kv_latent, k_pe_src = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_dim], dim=-1)
        k_pe = k_pe_src.view(bs, seq, 1, self.qk_rope_dim).transpose(1, 2)
        kv_latent = self.kv_a_layernorm(kv_latent)
        kv_b = self.kv_b_proj(kv_latent)
        kv = kv_b.view(bs, seq, self.num_heads, self.qk_nope_dim + self.v_head_dim).transpose(1, 2)
        k_nope, v = torch.split(kv, [self.qk_nope_dim, self.v_head_dim], dim=-1)

        if freqs_cis is None:
            raise RuntimeError(
                "DeepseekV2DecoderLayer requires freqs_cis from DeepseekV2Model (shared RoPE, HF-aligned)."
            )
        q_pe, k_pe = apply_rotary_emb(q_pe, k_pe, freqs_cis.to(device=q_pe.device))
        k_pe = k_pe.expand(bs, self.num_heads, seq, self.qk_rope_dim)

        query_states = q_nope.new_empty(bs, self.num_heads, seq, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_dim] = q_nope
        query_states[:, :, :, self.qk_nope_dim :] = q_pe

        key_states = k_pe.new_empty(bs, self.num_heads, seq, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_dim] = k_nope
        key_states[:, :, :, self.qk_nope_dim :] = k_pe

        scale = float(self.softmax_scale)

        if use_mla_cache and not is_prefill and bs == 1 and seq == 1:
            k_buf, v_buf = mla_kv[mla_layer_idx]
            L = mla_cached_len
            k_past = k_buf[:L]
            v_past = v_buf[:L]
            k_new = key_states[0, :, 0, :]
            v_new = v[0, :, 0, :]
            K_all = torch.cat([k_past, k_new.unsqueeze(0)], dim=0).transpose(0, 1).unsqueeze(0)
            V_all = torch.cat([v_past, v_new.unsqueeze(0)], dim=0).transpose(0, 1).unsqueeze(0)
            attn_scores = _qk_attn_scores(query_states, K_all, scale)
            attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_out = torch.matmul(attn_probs, V_all)
            attn_out = attn_out.transpose(1, 2).reshape(bs, seq, -1)
            k_buf[L].copy_(k_new)
            v_buf[L].copy_(v_new)
        else:
            attn_scores = _qk_attn_scores(query_states, key_states, scale)
            causal = torch.triu(
                torch.full((seq, seq), float("-inf"), device=attn_scores.device, dtype=attn_scores.dtype),
                diagonal=1,
            )
            attn_scores = attn_scores + causal
            attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_out = torch.matmul(attn_probs, v)
            attn_out = attn_out.transpose(1, 2).reshape(bs, seq, -1)
            if use_mla_cache and is_prefill:
                pr = md.get("mla_prefill_kv_range")
                if pr is not None:
                    s, e = int(pr[0]), int(pr[1])
                    k_buf, v_buf = mla_kv[mla_layer_idx]
                    chunk = e - s
                    k_buf[s:e].copy_(key_states[0, :, :chunk, :].permute(1, 0, 2))
                    v_buf[s:e].copy_(v[0, :, :chunk, :].permute(1, 0, 2))

        hidden_states = x + self.o_proj(attn_out)
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

class DeepseekV2Model(nn.Module):
    def __init__(self, config: LiteConfig, vllm_config: VllmConfig):
        super().__init__()
        self.config = config
        _normalize_deepseek_rope_config(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # One RoPE module for all layers (same as transformers DeepseekV2Model) — dynamic YaRN updates inv_freq once.
        self.rotary_emb = DeepseekV2RotaryEmbedding(config, device=None)
        self.layers = nn.ModuleList([DeepseekV2DecoderLayer(config, vllm_config.quant_config, prefix=f"model.layers.{i}") for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))
    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        x = self.embed_tokens(input_ids)
        position_ids = positions.long()
        # Match HF: rotary_emb(embed_out, position_ids) — x is only used for device/dtype in RoPE forward.
        freqs_cis = self.rotary_emb(x, position_ids)
        for i in range(len(self.layers)):
            layer_md = dict(attn_metadata) if isinstance(attn_metadata, dict) else attn_metadata
            if isinstance(layer_md, dict):
                layer_md = {**layer_md, "mla_layer_idx": i}
            x = self.layers[i](x, positions, kv_caches[i], layer_md, lora_mapping, freqs_cis=freqs_cis)
        return self.norm(x)

class DeepseekV2ForCausalLM(nn.Module):
    def __init__(self, vllm_config: VllmConfig):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        _normalize_deepseek_rope_config(self.config)
        self.model = DeepseekV2Model(self.config, vllm_config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata, lora_mapping)
        return self.lm_head(hidden_states)
