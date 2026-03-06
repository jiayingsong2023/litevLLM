# SPDX-License-Identifier: Apache-2.0
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Optional, List, Any, Tuple
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.layers.activation import Silu
from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe
from vllm.kernels.triton.gguf_dequant import gguf_dequantize
from vllm.model_executor.layers.quantization.gguf_kernels import ggml_dequantize_fallback


def _allow_numeric_clamp() -> bool:
    return os.environ.get("FASTINFERENCE_QWEN35_ALLOW_NUMERIC_CLAMP", "0") == "1"


_FINITE_CHECK_COUNTER = 0


def _ensure_finite_or_clamp(tensor: torch.Tensor, tensor_name: str) -> torch.Tensor:
    global _FINITE_CHECK_COUNTER
    check_interval = int(os.environ.get("FASTINFERENCE_QWEN35_FINITE_CHECK_INTERVAL", "1"))
    if check_interval <= 0:
        return tensor
    _FINITE_CHECK_COUNTER += 1
    if (_FINITE_CHECK_COUNTER % check_interval) != 0:
        return tensor
    if torch.isfinite(tensor).all():
        return tensor
    if _allow_numeric_clamp():
        return torch.nan_to_num(tensor, nan=0.0, posinf=1e4, neginf=-1e4)
    raise RuntimeError(
        f"{tensor_name} contains non-finite values. "
        "Set FASTINFERENCE_QWEN35_ALLOW_NUMERIC_CLAMP=1 only for debugging."
    )


class Qwen3_5Attention(nn.Module):
    def __init__(self, config, layer_id, quant_config, prefix, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.expected_full_qkv = self.q_size + 2 * self.kv_size
        self.expected_linear_qkv = self.hidden_size * 4
        if self.layer_type == "full_attention":
            self.qkv_proj = LiteLinear(
                config.hidden_size,
                self.expected_full_qkv,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.attn_qkv",
            )
            self.o_proj = LiteLinear(
                self.q_size,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.attn_output",
            )
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.qkv_proj = LiteLinear(
                config.hidden_size,
                self.expected_linear_qkv,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.attn_qkv",
            )
            self.attn_gate = LiteLinear(
                config.hidden_size,
                self.hidden_size * 2,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.attn_gate",
            )
            self.ssm_alpha = LiteLinear(
                config.hidden_size,
                32,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.ssm_alpha",
            )
            self.ssm_beta = LiteLinear(
                config.hidden_size,
                32,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.ssm_beta",
            )
            self.ssm_out = LiteLinear(
                self.hidden_size * 2,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.ssm_out",
            )
            self.ssm_state_a = nn.Parameter(torch.zeros(32, device="cuda", dtype=torch.float32), requires_grad=False)
            self.ssm_dt_bias = nn.Parameter(torch.zeros(32, device="cuda", dtype=torch.float32), requires_grad=False)
            self.ssm_norm_weight = nn.Parameter(torch.ones(128, device="cuda", dtype=torch.float32), requires_grad=False)
            self.ssm_conv1d_weight = nn.Parameter(
                torch.zeros(self.expected_linear_qkv, 4, device="cuda", dtype=torch.float16),
                requires_grad=False,
            )
            self.register_buffer("_ssm_state", torch.zeros(1, 32, device="cuda", dtype=torch.float32), persistent=False)
            self.register_buffer(
                "_conv_state",
                torch.zeros(1, 4, self.expected_linear_qkv, device="cuda", dtype=torch.float32),
                persistent=False,
            )
        self.scale = self.head_dim**-0.5

    def forward(self, hidden_states, positions, kv_cache, attn_metadata, lora_mapping=None):
        bsz, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states, lora_mapping=lora_mapping)
        if qkv.dim() == 2:
            qkv = qkv.unsqueeze(1)

        if self.layer_type == "full_attention":
            if qkv.shape[-1] < 2 * self.kv_size + self.q_size:
                raise RuntimeError(
                    f"Qwen3.5 full-attention qkv too small: {qkv.shape[-1]}."
                )
            q_part_size = qkv.shape[-1] - 2 * self.kv_size
            q_part, k, v = qkv.split([q_part_size, self.kv_size, self.kv_size], dim=-1)
            gate = None
            if q_part_size == self.q_size * 2:
                q, gate = torch.chunk(q_part, 2, dim=-1)
            elif q_part_size == self.q_size:
                q = q_part
            else:
                raise RuntimeError(
                    f"Unsupported full-attention q part size: {q_part_size}."
                )
            q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(bsz, 1, -1)
            k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(bsz, 1, -1)
            from vllm.attention.ops.triton_paged_attn import triton_paged_attention

            output = triton_paged_attention(
                q[:, -1:, :].view(bsz, self.num_heads, self.head_dim),
                k[:, -1:, :].view(bsz, self.num_kv_heads, self.head_dim),
                v[:, -1:, :].view(bsz, self.num_kv_heads, self.head_dim),
                kv_cache,
                attn_metadata["slot_mapping"],
                attn_metadata["seq_lens"],
                None,
                self.scale,
            )
            attn_hidden = output.view(bsz, 1, -1)
            if gate is not None:
                attn_hidden = attn_hidden * torch.sigmoid(gate[:, -1:, :])
            return _ensure_finite_or_clamp(
                self.o_proj(attn_hidden, lora_mapping=lora_mapping),
                "qwen3_5.full_attention.output",
            )

        if qkv.shape[-1] < self.q_size + 2:
            raise RuntimeError(
                f"Qwen3.5 linear-attention qkv too small: {qkv.shape[-1]} for q_size={self.q_size}."
            )

        linear_q_size = self.q_size
        linear_remaining = qkv.shape[-1] - linear_q_size
        linear_k_size = linear_remaining // 2
        linear_v_size = linear_remaining - linear_k_size
        target_size = linear_k_size + linear_v_size
        hidden_last = hidden_states[:, -1, :]

        if self._ssm_state.shape[0] != bsz:
            self._ssm_state = torch.zeros(
                bsz,
                self._ssm_state.shape[-1],
                device=hidden_states.device,
                dtype=torch.float32,
            )
        kernel_size = int(self.ssm_conv1d_weight.shape[-1])
        if (
            self._conv_state.shape[0] != bsz
            or self._conv_state.shape[1] != kernel_size
            or self._conv_state.shape[2] != qkv.shape[-1]
        ):
            self._conv_state = torch.zeros(
                bsz,
                kernel_size,
                qkv.shape[-1],
                device=hidden_states.device,
                dtype=torch.float32,
            )

        mixed_qkv = qkv[:, -1, :].float()
        self._conv_state = torch.roll(self._conv_state, shifts=-1, dims=1)
        self._conv_state[:, -1, :] = mixed_qkv
        conv_kernel = self.ssm_conv1d_weight.float()
        if conv_kernel.shape[-1] != self._conv_state.shape[1]:
            conv_out = mixed_qkv
        else:
            conv_kernel = conv_kernel.to(self._conv_state.dtype)
            conv_out = torch.einsum("bkd,dk->bd", self._conv_state, conv_kernel)

        core_base = conv_out[:, linear_q_size: linear_q_size + target_size]
        k_proj = core_base[:, :linear_k_size]
        v_proj = core_base[:, linear_k_size:]

        a = self.ssm_alpha(hidden_last).float()
        b = self.ssm_beta(hidden_last).float()
        g = -torch.exp(self.ssm_state_a).unsqueeze(0) * F.softplus(
            a + self.ssm_dt_bias.unsqueeze(0)
        )
        beta = torch.sigmoid(b)
        ssm_channels = self._ssm_state.shape[-1]
        group_width = max(1, linear_v_size // ssm_channels)
        usable_v = group_width * ssm_channels
        usable_k = group_width * ssm_channels
        v_group = v_proj[:, :usable_v].view(bsz, ssm_channels, group_width).mean(dim=-1)
        k_group = k_proj[:, :usable_k].view(bsz, ssm_channels, group_width).mean(dim=-1)
        self._ssm_state = torch.exp(g) * self._ssm_state + beta * v_group
        state_signal = self._ssm_state * torch.sigmoid(k_group)
        repeat_factor = max(1, target_size // ssm_channels)
        expanded_state = state_signal.repeat_interleave(repeat_factor, dim=-1)
        if expanded_state.shape[-1] < target_size:
            expanded_state = F.pad(expanded_state, (0, target_size - expanded_state.shape[-1]))
        expanded_state = expanded_state[:, :target_size]

        core = core_base + expanded_state
        core_rms = torch.rsqrt(core.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        core = core * core_rms
        output_gate = torch.sigmoid(self.attn_gate(hidden_last)).float()
        core = core * output_gate
        norm_scale = self.ssm_norm_weight.mean()
        core = (core * norm_scale).to(hidden_states.dtype)
        linear_out = self.ssm_out(core).view(bsz, 1, -1)
        return _ensure_finite_or_clamp(linear_out, "qwen3_5.linear_attention.output")

class Qwen3_5MLP(nn.Module):
    def __init__(self, config, quant_config, prefix):
        super().__init__()
        self.gate_up_proj = LiteLinear(config.hidden_size, 2 * config.intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.ffn_gate_up")
        self.down_proj = LiteLinear(config.intermediate_size, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.ffn_down")
        self.act = Silu()
    def forward(self, x, lora_mapping=None):
        if x.dim() == 2: x = x.unsqueeze(1)
        g, u = self.gate_up_proj(x[:, -1:, :], lora_mapping=lora_mapping).chunk(2, dim=-1)
        return self.down_proj(self.act(g) * u, lora_mapping=lora_mapping)

class Qwen3_5Layer(nn.Module):
    def __init__(self, config, layer_id, quant_config=None, prefix=""):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        layer_types = getattr(config, "layer_types", None)
        layer_type = "full_attention"
        if isinstance(layer_types, list) and layer_id < len(layer_types):
            layer_type = layer_types[layer_id]
        self.self_attn = Qwen3_5Attention(config, layer_id, quant_config, prefix, layer_type=layer_type)
        if hasattr(config, "num_experts") and config.num_experts > 0: self.mlp = Qwen3_5MoE(config, quant_config, prefix)
        else: self.mlp = Qwen3_5MLP(config, quant_config, prefix)
    def forward(self, hidden_states, positions, kv_cache, attn_metadata, lora_mapping=None):
        h = self.input_layernorm(hidden_states)
        attn_res = self.self_attn(h, positions, kv_cache, attn_metadata, lora_mapping=lora_mapping)
        attn_res = attn_res.to(hidden_states.dtype)
        if hidden_states.dim() == 3: hidden_states = hidden_states[:, -1:, :] + attn_res
        else: hidden_states = hidden_states + attn_res.squeeze(1)
        hidden_states = hidden_states.to(self.post_attention_layernorm.weight.dtype)
        h = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + self.mlp(h, lora_mapping=lora_mapping).to(hidden_states.dtype)
        return hidden_states

class Qwen3_5Model(nn.Module):
    def __init__(self, vllm_config, prefix=""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3_5Layer(config, i, quant_config=vllm_config.quant_config, prefix=f"blk.{i}") for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        x = self.embed_tokens(input_ids)
        for i in range(len(self.layers)): x = self.layers[i](x, positions, kv_caches[i], attn_metadata, lora_mapping=lora_mapping)
        return self.norm(x)

class Qwen3_5ForCausalLM(nn.Module):
    def __init__(self, vllm_config, prefix=""):
        super().__init__()
        self.model = Qwen3_5Model(vllm_config, prefix)
        self.lm_head = LiteLinear(vllm_config.model_config.hf_config.hidden_size, vllm_config.model_config.hf_config.vocab_size, bias=False, prefix="output")
    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        return self.lm_head(self.model(input_ids, positions, kv_caches, attn_metadata, lora_mapping=lora_mapping), lora_mapping=lora_mapping)

class Qwen3_5MoE(nn.Module):
    def __init__(self, config, quant_config, prefix):
        super().__init__()
        self.prefix = prefix
        self.num_experts = config.num_experts; self.topk = config.num_experts_per_tok
        self.w1 = LiteLinear(config.hidden_size, self.num_experts * config.moe_intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.ffn_gate_exps")
        self.w1_up = LiteLinear(config.hidden_size, self.num_experts * config.moe_intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.ffn_up_exps")
        self.w2 = LiteLinear(config.moe_intermediate_size, self.num_experts * config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.ffn_down_exps")
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False).cuda().half()
        self._expert_triplet_cache: "OrderedDict[tuple[int, torch.dtype], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]" = OrderedDict()
        self._max_expert_cache_size = int(os.environ.get("FASTINFERENCE_QWEN35_MOE_CACHE_SIZE", "192"))
        self._prewarm_budget = int(os.environ.get("FASTINFERENCE_QWEN35_MOE_PREWARM", str(max(self.topk * 8, 32))))
        self._prewarm_done = False
        self._grouped_moe_enabled = os.environ.get("FASTINFERENCE_QWEN35_GROUPED_MOE", "1") == "1"
        self._grouped_moe_fallback = os.environ.get("FASTINFERENCE_QWEN35_GROUPED_MOE_FALLBACK", "1") == "1"
        self._grouped_moe_min_tokens = int(os.environ.get("FASTINFERENCE_QWEN35_GROUPED_MOE_MIN_TOKENS", "2"))

    def _dequantize_expert_matrix(self, layer: LiteLinear, expert_idx: int, dtype: torch.dtype) -> torch.Tensor:
        if getattr(layer, "qweight", None) is None:
            raise RuntimeError(f"Expert tensor not loaded for layer '{self.prefix}', expert={expert_idx}")
        if getattr(layer, "gguf_quant_type", None) is None or getattr(layer, "gguf_shape", None) is None:
            raise RuntimeError(
                f"GGUF quant metadata missing for layer '{self.prefix}', expert={expert_idx}"
            )
        packed = layer.qweight[expert_idx].contiguous()
        m_dim = int(layer.gguf_shape[1])
        n_dim = int(layer.gguf_shape[0])
        matrix = ggml_dequantize_fallback(
            packed,
            int(layer.gguf_quant_type),
            m_dim,
            n_dim,
            dtype,
        )
        return matrix

    def _get_expert_triplet(
        self, expert_idx: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_key = (expert_idx, dtype)
        cached = self._expert_triplet_cache.get(cache_key)
        if cached is not None:
            self._expert_triplet_cache.move_to_end(cache_key)
            return cached

        gate_w = self._dequantize_expert_matrix(self.w1, expert_idx, dtype)
        up_w = self._dequantize_expert_matrix(self.w1_up, expert_idx, dtype)
        down_w = self._dequantize_expert_matrix(self.w2, expert_idx, dtype)
        triplet = (gate_w, up_w, down_w)
        self._expert_triplet_cache[cache_key] = triplet
        if len(self._expert_triplet_cache) > self._max_expert_cache_size:
            self._expert_triplet_cache.popitem(last=False)
        return triplet

    def _prewarm_hot_experts(self, router_logits: torch.Tensor, dtype: torch.dtype) -> None:
        if self._prewarm_done:
            return
        prewarm_k = min(self.num_experts, max(self.topk, self._prewarm_budget))
        mean_router = router_logits.float().mean(dim=0)
        _, warm_ids = torch.topk(mean_router, k=prewarm_k, dim=-1)
        for expert_id in warm_ids.tolist():
            self._get_expert_triplet(int(expert_id), dtype)
        self._prewarm_done = True

    def _forward_token_loop(
        self,
        curr_x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(curr_x)
        for token_idx in range(curr_x.shape[0]):
            token_hidden = curr_x[token_idx : token_idx + 1]
            expert_indices = [int(expert_id.item()) for expert_id in topk_ids[token_idx]]
            triplets = [self._get_expert_triplet(expert_id, curr_x.dtype) for expert_id in expert_indices]
            gate_w = torch.stack([triplet[0] for triplet in triplets], dim=0)
            up_w = torch.stack([triplet[1] for triplet in triplets], dim=0)
            down_w = torch.stack([triplet[2] for triplet in triplets], dim=0)
            hidden_vec = token_hidden.squeeze(0)
            gate_act = torch.einsum("kmh,h->km", gate_w, hidden_vec)
            up_act = torch.einsum("kmh,h->km", up_w, hidden_vec)
            mixed = F.silu(gate_act) * up_act
            expert_out = torch.einsum("km,khm->kh", mixed, down_w)
            token_out = (expert_out * topk_weights[token_idx].unsqueeze(-1)).sum(dim=0, keepdim=True)
            output[token_idx : token_idx + 1] = _ensure_finite_or_clamp(
                token_out,
                "qwen3_5.moe.token_output",
            )
        return output

    def _forward_grouped(
        self,
        curr_x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = curr_x.shape[0]
        flat_expert_ids = topk_ids.reshape(-1)
        flat_token_ids = torch.arange(
            num_tokens,
            device=curr_x.device,
            dtype=torch.long,
        ).repeat_interleave(self.topk)
        flat_route_weights = topk_weights.reshape(-1).to(curr_x.dtype)

        unique_expert_ids = torch.unique(flat_expert_ids, sorted=True)
        output = torch.zeros_like(curr_x)

        for expert_id_tensor in unique_expert_ids:
            expert_id = int(expert_id_tensor.item())
            selected_mask = flat_expert_ids == expert_id_tensor
            if not torch.any(selected_mask):
                continue
            token_indices = flat_token_ids[selected_mask]
            route_weights = flat_route_weights[selected_mask].unsqueeze(-1)
            hidden_batch = curr_x.index_select(0, token_indices)
            gate_w, up_w, down_w = self._get_expert_triplet(expert_id, curr_x.dtype)

            # Group tokens per expert to reduce tiny matmul launches.
            gate_act = hidden_batch @ gate_w.transpose(0, 1)
            up_act = hidden_batch @ up_w.transpose(0, 1)
            mixed = F.silu(gate_act) * up_act
            expert_out = mixed @ down_w.transpose(0, 1)
            weighted_out = _ensure_finite_or_clamp(
                expert_out * route_weights,
                "qwen3_5.moe.grouped_weighted_output",
            )
            output.index_add_(0, token_indices, weighted_out)

        return output

    def forward(self, x, lora_mapping=None):
        if x.dim() == 2: x = x.unsqueeze(1)
        curr_x = x[:, -1:, :].view(-1, x.shape[-1])
        gate_input = curr_x.to(self.gate.weight.dtype)
        router_logits = self.gate(gate_input).to(curr_x.dtype)
        self._prewarm_hot_experts(router_logits, curr_x.dtype)
        topk_vals, topk_ids = torch.topk(router_logits, k=self.topk, dim=-1)
        topk_weights = torch.softmax(topk_vals, dim=-1)
        use_grouped = self._grouped_moe_enabled and curr_x.shape[0] >= self._grouped_moe_min_tokens
        if use_grouped:
            try:
                output = self._forward_grouped(curr_x, topk_ids, topk_weights)
            except RuntimeError:
                if not self._grouped_moe_fallback:
                    raise
                output = self._forward_token_loop(curr_x, topk_ids, topk_weights)
        else:
            output = self._forward_token_loop(curr_x, topk_ids, topk_weights)
        return output.view(x.shape[0], 1, -1)

class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLM): pass
class Qwen3_5MoeForConditionalGeneration(Qwen3_5ForCausalLM): pass
class Qwen3_5ForConditionalGeneration(Qwen3_5ForCausalLM): pass
