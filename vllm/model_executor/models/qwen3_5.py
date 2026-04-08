# SPDX-License-Identifier: Apache-2.0
# GatedDeltaNet helpers below align with Hugging Face `modeling_qwen3_5.py` (torch fallback path).
import os
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Any, Tuple
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.layers.quantization.tensor import AWQWeight, PackedInt4Weight
from .lite_config import LiteConfig
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
from vllm.model_executor.moe_fp8_utils import (
    dims_ok_for_moe_fp8,
    fp8_scale_shape_2d,
    moe_fp8_enabled,
    moe_offload_enabled,
    moe_expert_lru_size,
    moe_fp8_dequant_to_linear_weight,
)


def _env_truthy(name: str) -> bool:
    """True for 1/true/yes/on (case-insensitive). Used for debug / ablation toggles."""
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _env_qwen35_sdpa_prefill_enabled() -> bool:
    """
    When ON: full-attention prefill uses ``scaled_dot_product_attention`` (faster, may differ slightly from HF).
    Default OFF: prefill uses HF ``eager_attention_forward`` math (repeat_kv + float32 softmax + causal mask).
    Set FASTINFERENCE_QWEN35_FULLATTN_USE_SDP_PREFILL=1 to enable SDPA (e2e_full_benchmark sets this for throughput).
    """
    v = os.environ.get("FASTINFERENCE_QWEN35_FULLATTN_USE_SDP_PREFILL", "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def _repeat_kv_hf(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Match ``transformers.models.qwen3_5.modeling_qwen3_5.repeat_kv`` (GQA broadcast)."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    x = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return x.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _hf_eager_causal_attention_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """
    Match HF ``eager_attention_forward`` for causal prefill: GQA repeat_kv, matmul logits,
    upper-triangular mask, softmax in float32, then matmul with values.
    Shapes: query (B, nh, S, hd); key/value (B, nkv, S, hd). Returns (B, S, nh, hd).
    """
    n_rep = query.shape[1] // key.shape[1]
    key_states = _repeat_kv_hf(key, n_rep)
    value_states = _repeat_kv_hf(value, n_rep)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    seq_len = query.shape[2]
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool), diagonal=1)
    attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output.transpose(1, 2).contiguous()


def _hf_conv_state_from_preconv(mixed_b_c_l: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Match HF ``F.pad(mixed_qkv, (k - L, 0))`` on the sequence dimension (last).
    Positive pad left-pads when L < k; negative pad crops to the last k positions when L >= k.
    """
    L = mixed_b_c_l.shape[-1]
    k = kernel_size
    pad_left = k - L
    if pad_left >= 0:
        return F.pad(mixed_b_c_l, (pad_left, 0))
    return mixed_b_c_l[..., -k:].contiguous()


def _torch_causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    conv1d: nn.Conv1d,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Match HF ``torch_causal_conv1d_update`` (pre-conv layout ``B, C, L``).
    Returns ``(silu(conv_out)`` sliced to current ``L``, ``new_conv_state``).
    """
    weight = conv1d.weight
    bias = conv1d.bias
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    new_state = hidden_states_new[:, :, -state_len:].contiguous()
    out = F.conv1d(hidden_states_new, weight, bias, stride=1, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    return out.to(hidden_states.dtype), new_state


def _env_qwen35_fla_chunk_enabled() -> bool:
    """
    Default ON (CUDA): use FLA fused ``chunk_gated_delta_rule`` to match Hugging Face Qwen3.5 text.
    Set FASTINFERENCE_QWEN35_USE_FLA_CHUNK=0 to force the pure PyTorch reference (slower, drifts vs HF).
    """
    v = os.environ.get("FASTINFERENCE_QWEN35_USE_FLA_CHUNK", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


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
    """
    Pure PyTorch chunk gated delta rule (reference path).

    Numerically matches ``fla.ops.gated_delta_rule.naive.naive_chunk_gated_delta_rule``
    (flash-linear-attention reference) for equal inputs on CPU/GPU.

    Set ``FASTINFERENCE_QWEN35_USE_FLA_CHUNK=1`` to use the fused ``fla`` kernel on CUDA
    when ``flash-linear-attention`` is installed (optional HF-aligned fast path).
    """
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


def _hf_torch_recurrent_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool = True,
):
    """
    Match HF ``torch_recurrent_gated_delta_rule`` (decode / streaming one step per call).
    Used when ``seq_len == 1`` with a non-None recurrent state; chunk+cumsum path differs for single-step decode.
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def _chunk_gated_delta_rule_backend(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    force_torch: bool = False,
):
    """
    Dispatch: FLA fused kernel on CUDA when enabled (default: on, matches HF), else PyTorch reference.
    ``force_torch=True`` (LiteEngine streaming): always use PyTorch chunk so recurrent decode state matches.
    """
    if (
        not force_torch
        and _env_qwen35_fla_chunk_enabled()
        and query.device.type == "cuda"
    ):
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        except ImportError:
            import warnings

            warnings.warn(
                "FASTINFERENCE_QWEN35_USE_FLA_CHUNK is enabled but flash-linear-attention is not installed; "
                "falling back to the pure PyTorch chunk rule.",
                stacklevel=2,
            )
        else:
            head_dim = query.shape[-1]
            scale = head_dim ** (-0.5)
            return chunk_gated_delta_rule(
                query,
                key,
                value,
                g,
                beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
    return _torch_chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        chunk_size=chunk_size,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


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
    Matches Hugging Face ``Qwen3_5RMSNorm``: ``(1 + weight)`` scale with ``weight`` initialized to zeros.
    Same checkpoint tensors as HF / safetensors; GGUF exports that follow HF naming use the same delta weights.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


def qwen35_layer_type(layer_idx: int, config: LiteConfig) -> str:
    """Match HF ``Qwen3_5MoeTextConfig`` / dense ``Qwen3_5TextConfig`` layer_types pattern."""
    lt = getattr(config, "layer_types", None)
    if isinstance(lt, list) and layer_idx < len(lt):
        return str(lt[layer_idx])
    interval = int(getattr(config, "full_attention_interval", 4) or 4)
    return "linear_attention" if bool((layer_idx + 1) % interval) else "full_attention"


class Qwen3_5MoeTopKRouterLite(nn.Module):
    """Aligned with HF ``Qwen3_5MoeTopKRouter``."""

    def __init__(self, config: LiteConfig):
        super().__init__()
        self.top_k = int(config.num_experts_per_tok)
        self.num_experts = int(config.num_experts)
        self.hidden_dim = int(config.hidden_size)
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        router_logits = F.softmax(router_logits, dtype=torch.float32, dim=-1)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        router_scores = router_top_value
        return router_logits, router_scores, router_indices


class Qwen3_5MoeExpertsLite(nn.Module):
    """Aligned with HF ``Qwen3_5MoeExperts`` (loop over hit experts).

    Optional GGUF packed uint8 (FASTINFERENCE_MOE_PACKED_GGUF=1): load-time RSS avoids full-blob dequant.
    Optional FP8 block weights (FASTINFERENCE_MOE_FP8=1) halve expert weight bytes (incompatible with packed).
    Optional CPU offload + GPU LRU (FASTINFERENCE_MOE_OFFLOAD=1) lowers GPU residency.
    """

    def __init__(self, config: LiteConfig):
        super().__init__()
        self.num_experts = int(config.num_experts)
        self.hidden_dim = int(config.hidden_size)
        self.intermediate_dim = int(config.moe_intermediate_size)
        f8 = torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else None
        
        self.fp8_moe = bool(
            f8 is not None
            and moe_fp8_enabled()
            and dims_ok_for_moe_fp8(self.hidden_dim, self.intermediate_dim)
        )
        self.moe_cpu_offload = bool(
            self.fp8_moe and moe_offload_enabled()
        )
        self.lru_size = moe_expert_lru_size()
        self._lru_gpu: "OrderedDict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]" = (
            OrderedDict()
        )

        # Standard ParameterLists for experts - populated during load
        self.gate_up_proj = nn.ParameterList([
            nn.Parameter(torch.empty(0)) for _ in range(self.num_experts)
        ])
        self.down_proj = nn.ParameterList([
            nn.Parameter(torch.empty(0)) for _ in range(self.num_experts)
        ])

        if self.fp8_moe and self.moe_cpu_offload:
            ou, inn = fp8_scale_shape_2d(2 * self.intermediate_dim, self.hidden_dim)
            ou_d, inn_d = fp8_scale_shape_2d(self.hidden_dim, self.intermediate_dim)
            self.register_buffer(
                "_gate_up_fp8_cpu",
                torch.empty(
                    self.num_experts,
                    2 * self.intermediate_dim,
                    self.hidden_dim,
                    dtype=f8,
                ).pin_memory(),
                persistent=True,
            )
            self.register_buffer(
                "_gate_up_scale_cpu",
                torch.empty(self.num_experts, ou, inn, dtype=torch.float32).pin_memory(),
                persistent=True,
            )
            self.register_buffer(
                "_down_fp8_cpu",
                torch.empty(
                    self.num_experts,
                    self.hidden_dim,
                    self.intermediate_dim,
                    dtype=f8,
                ).pin_memory(),
                persistent=True,
            )
            self.register_buffer(
                "_down_scale_cpu",
                torch.empty(self.num_experts, ou_d, inn_d, dtype=torch.float32).pin_memory(),
                persistent=True,
            )
            # Placeholders so load / state_dict can target stable keys (overwritten by loader).
            self.register_buffer("_gate_up_proj_placeholder", torch.empty(0), persistent=False)
            self.register_buffer("_down_proj_placeholder", torch.empty(0), persistent=False)
        elif self.fp8_moe:
            ou, inn = fp8_scale_shape_2d(2 * self.intermediate_dim, self.hidden_dim)
            ou_d, inn_d = fp8_scale_shape_2d(self.hidden_dim, self.intermediate_dim)
            self.gate_up_proj = nn.Parameter(
                torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim, dtype=f8)
            )
            self.gate_up_scale = nn.Parameter(
                torch.empty(self.num_experts, ou, inn, dtype=torch.float32)
            )
            self.down_proj = nn.Parameter(
                torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim, dtype=f8)
            )
            self.down_scale = nn.Parameter(torch.empty(self.num_experts, ou_d, inn_d, dtype=torch.float32))
        else:
            self.gate_up_proj = nn.Parameter(
                torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
            )
            self.down_proj = nn.Parameter(
                torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
            )

    def _apply(self, fn):
        # Keep pinned CPU expert buffers on CPU when moving the rest of the model to CUDA.
        skip = {"_gate_up_fp8_cpu", "_gate_up_scale_cpu", "_down_fp8_cpu", "_down_scale_cpu"}
        for module in self.children():
            module._apply(fn)
        for name, param in self._parameters.items():
            if param is not None:
                self._parameters[name] = fn(param)
        for name, buf in self._buffers.items():
            if name in skip:
                continue
            if buf is not None:
                self._buffers[name] = fn(buf)
        return self

    def _lru_get_expert_fp8_gpu(
        self, expert_idx: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if expert_idx in self._lru_gpu:
            self._lru_gpu.move_to_end(expert_idx)
            return self._lru_gpu[expert_idx]
        g = self._gate_up_fp8_cpu[expert_idx].to(device=device, non_blocking=True)
        gs = self._gate_up_scale_cpu[expert_idx].to(device=device, non_blocking=True)
        d = self._down_fp8_cpu[expert_idx].to(device=device, non_blocking=True)
        ds = self._down_scale_cpu[expert_idx].to(device=device, non_blocking=True)
        tpl = (g, gs, d, ds)
        self._lru_gpu[expert_idx] = tpl
        if len(self._lru_gpu) > self.lru_size:
            self._lru_gpu.popitem(last=False)
        return tpl

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        dev = hidden_states.device
        hid_dtype = hidden_states.dtype
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            if self.moe_cpu_offload:
                g_fp8, g_s, d_fp8, d_s = self._lru_get_expert_fp8_gpu(int(expert_idx), dev)
                w_g = moe_fp8_dequant_to_linear_weight(g_fp8, g_s, hid_dtype)
                gate, up = F.linear(current_state, w_g).chunk(2, dim=-1)
                current_hidden_states = F.silu(gate) * up
                w_d = moe_fp8_dequant_to_linear_weight(d_fp8, d_s, hid_dtype)
                current_hidden_states = F.linear(current_hidden_states, w_d)
            elif self.fp8_moe:
                w_g = moe_fp8_dequant_to_linear_weight(
                    self.gate_up_proj[expert_idx], self.gate_up_scale[expert_idx], hid_dtype
                )
                gate, up = F.linear(current_state, w_g).chunk(2, dim=-1)
                current_hidden_states = F.silu(gate) * up
                w_d = moe_fp8_dequant_to_linear_weight(
                    self.down_proj[expert_idx], self.down_scale[expert_idx], hid_dtype
                )
                current_hidden_states = F.linear(current_hidden_states, w_d)
            else:
                w_gu = self.gate_up_proj[expert_idx].to(device=dev, dtype=hid_dtype)
                gate, up = F.linear(current_state, w_gu).chunk(2, dim=-1)
                current_hidden_states = F.silu(gate) * up
                w_d = self.down_proj[expert_idx].to(device=dev, dtype=hid_dtype)
                current_hidden_states = F.linear(current_hidden_states, w_d)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class Qwen3_5MoeSparseMoeBlock(nn.Module):
    """Aligned with HF ``Qwen3_5MoeSparseMoeBlock`` (shared expert order + gate scaling)."""

    def __init__(self, config: LiteConfig, quant_config, prefix: str = ""):
        super().__init__()
        self.config = config
        self.gate = Qwen3_5MoeTopKRouterLite(config)
        self.experts = Qwen3_5MoeExpertsLite(config)
        sh = int(config.shared_expert_intermediate_size)
        self.shared_expert = nn.Module()
        self.shared_expert.gate_proj = LiteLinear(
            config.hidden_size, sh, bias=False, quant_config=quant_config, prefix=f"{prefix}.shared_expert.gate_proj"
        )
        self.shared_expert.up_proj = LiteLinear(
            config.hidden_size, sh, bias=False, quant_config=quant_config, prefix=f"{prefix}.shared_expert.up_proj"
        )
        self.shared_expert.down_proj = LiteLinear(
            sh, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.shared_expert.down_proj"
        )
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        shared_expert_output = self.shared_expert.down_proj(
            F.silu(self.shared_expert.gate_proj(hidden_states_reshaped)) * self.shared_expert.up_proj(hidden_states_reshaped)
        )
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        expert_output = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
        shared_expert_output = torch.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output
        expert_output = expert_output + shared_expert_output
        return expert_output.reshape(batch_size, sequence_length, hidden_dim)


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


def _qwen35_try_fused_awq_pair_matmul(
    x: torch.Tensor,
    lin_a: LiteLinear,
    lin_b: LiteLinear,
    owner: nn.Module,
    cache_key: str,
) -> Optional[torch.Tensor]:
    """
    Single quantized matmul for two LiteLinear layers with identical packed layout
    (same qweight/scales row shape) and same input/output sizes.

    Supports:
    - Classic AWQ (qweight + scales + qzeros): stacked AWQWeight.
    - Symmetric packed int4 (qweight + scales, qzeros None): stacked PackedInt4Weight (Qwen3.5 AWQ).

    Returns tensor (..., 2 * out_dim) or None to fall back to two forwards.
    """
    if lin_a.input_size != lin_b.input_size or lin_a.output_size != lin_b.output_size:
        return None
    ba = getattr(lin_a, "bias", None)
    bb = getattr(lin_b, "bias", None)
    fused_bias: Optional[torch.Tensor] = None
    if ba is not None and bb is not None:
        fused_bias = torch.cat([ba.reshape(-1), bb.reshape(-1)], dim=0).contiguous()
    elif ba is not None or bb is not None:
        return None
    if getattr(lin_a, "force_high_fidelity_awq", False) or getattr(
        lin_b, "force_high_fidelity_awq", False
    ):
        return None
    qwa = getattr(lin_a, "qweight", None)
    qwb = getattr(lin_b, "qweight", None)
    if qwa is None or qwb is None or qwa.numel() <= 1 or qwb.numel() <= 1:
        return None
    if tuple(qwa.shape) != tuple(qwb.shape):
        return None
    if tuple(lin_a.scales.shape) != tuple(lin_b.scales.shape):
        return None
    gs = int(getattr(lin_a, "group_size", 128))
    if int(getattr(lin_b, "group_size", 128)) != gs:
        return None

    za = getattr(lin_a, "qzeros", None)
    zb = getattr(lin_b, "qzeros", None)
    has_awq_zeros = (
        za is not None
        and zb is not None
        and za.numel() > 1
        and zb.numel() > 1
        and tuple(za.shape) == tuple(zb.shape)
    )
    has_symmetric_packed = (za is None or za.numel() <= 1) and (zb is None or zb.numel() <= 1)
    if not has_awq_zeros and not has_symmetric_packed:
        return None

    cache_attr = f"_fused_awq_pair_{cache_key}"
    fused_w = getattr(owner, cache_attr, None)
    if fused_w is None:
        q_cat = torch.cat([lin_a.qweight, lin_b.qweight], dim=0).contiguous()
        s_cat = torch.cat([lin_a.scales, lin_b.scales], dim=0).contiguous()
        if has_awq_zeros:
            fused_w = AWQWeight(
                q_cat,
                s_cat,
                torch.cat([lin_a.qzeros, lin_b.qzeros], dim=0).contiguous(),
                group_size=gs,
                prefix=getattr(lin_a, "prefix", "") or "fused_pair",
                high_fidelity=False,
                profile_hint=str(getattr(lin_a, "awq_profile_hint", "") or ""),
            )
        else:
            fused_w = PackedInt4Weight(
                q_cat,
                s_cat,
                group_size=gs,
                original_shape=getattr(lin_a, "weight_shape", None),
                prefix=getattr(lin_a, "prefix", "") or "fused_pair",
                high_fidelity=False,
                profile_hint=str(getattr(lin_a, "awq_profile_hint", "") or ""),
            )
        setattr(owner, cache_attr, fused_w)

    lead_shape = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    out2 = fused_w.matmul(x2, fused_bias)
    od = int(lin_a.output_size)
    return out2.reshape(*lead_shape, 2 * od)


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

    def __init__(
        self, config: LiteConfig, quant_config, prefix="", layer_idx: int = 0, use_moe: bool = False
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self._use_moe = use_moe
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
        if use_moe:
            self.mlp = Qwen3_5MoeSparseMoeBlock(config, quant_config, prefix=f"{prefix}.mlp")
        else:
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
        inf_config = attn_metadata.get("config") if isinstance(attn_metadata, dict) else getattr(attn_metadata, "config", None)
        fusion_level = inf_config.fusion_level if inf_config else 2

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
            conv_dtype = self.linear_attn.conv1d.weight.dtype
            # (B, C, L) pre-conv, same layout as HF after ``in_proj_qkv`` + transpose(1,2)
            x_legacy = mixed.transpose(1, 2).contiguous().to(dtype=conv_dtype)
            prev_cc = (
                conv_carry_list[self.layer_idx]
                if self.layer_idx < len(conv_carry_list)
                else None
            )
            if prev_cc is None:
                # No prior state: same as HF when ``causal_conv1d_fn`` is None — full ``nn.Conv1d`` + trim.
                mixed_conv = F.silu(self.linear_attn.conv1d(x_legacy)[:, :, :seq_len])
                mixed_qkv = mixed_conv.transpose(1, 2)
                new_conv_state = _hf_conv_state_from_preconv(x_legacy, k)
            else:
                # Chunk continuation or decode: HF ``torch_causal_conv1d_update`` (cat state + conv).
                mixed_conv, new_conv_state = _torch_causal_conv1d_update(
                    x_legacy, prev_cc, self.linear_attn.conv1d
                )
                mixed_qkv = mixed_conv.transpose(1, 2)
            if self.layer_idx < len(conv_carry_list):
                conv_carry_list[self.layer_idx] = new_conv_state.detach()
        else:
            mixed_qkv = mixed.transpose(1, 2)
            # Standard conv1d with padding=k-1 and causal slice [:seq_len]
            mixed_qkv = F.silu(self.linear_attn.conv1d(mixed_qkv)[:, :, :seq_len])
            mixed_qkv = mixed_qkv.transpose(1, 2)

        z = self.linear_attn.in_proj_z(h)
        z = z.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        if fusion_level >= 1: # Basic fusion (AB)
            ab = _qwen35_try_fused_awq_pair_matmul(
                h,
                self.linear_attn.in_proj_a,
                self.linear_attn.in_proj_b,
                self,
                "linear_ab",
            )
            if ab is not None:
                a, b = ab.split(self.num_v_heads, dim=-1)
            else:
                b = self.linear_attn.in_proj_b(h)
                a = self.linear_attn.in_proj_a(h)
        else:
            b = self.linear_attn.in_proj_b(h)
            a = self.linear_attn.in_proj_a(h)

        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )

        # Use -1 for sequence length to support Chunked Prefill robustness
        query = query.reshape(batch_size, -1, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, -1, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, -1, self.num_v_heads, self.head_v_dim)

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

        # HF uses ``torch_recurrent_gated_delta_rule`` for decode (seq==1, cached state), not chunk+cumsum.
        use_recurrent_decode = (
            use_stream
            and seq_len == 1
            and incoming_state is not None
        )
        if use_recurrent_decode:
            core_attn_out, last_state = _hf_torch_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=incoming_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_state = _chunk_gated_delta_rule_backend(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=incoming_state,
                output_final_state=use_stream,
                use_qk_l2norm_in_kernel=True,
                force_torch=use_stream,
            )
        if (
            use_stream
            and attn_carry_list is not None
            and self.layer_idx < len(attn_carry_list)
        ):
            attn_carry_list[self.layer_idx] = (
                None if last_state is None else last_state.detach()
            )

        # core_attn_out is [bs, seq, nh_v, hd_v] from rule
        # Normalize and gate PER HEAD (GGUF ssm_norm is [128])
        # [bs, seq, nh_v, hd_v] -> [bs, nh_v, seq, hd_v] for per-head norm
        core_heads = core_attn_out.transpose(1, 2)
        z_heads = z.transpose(1, 2)
        
        # Apply norm per-head. ssm_norm weight [128] will broadcast over [bs, nh_v, seq, 128]
        # correctly if the norm treats the last dimension as feature dimension.
        core_normed_heads = self.linear_attn.norm(core_heads, z_heads)
        
        # [bs, nh_v, seq, hd_v] -> [bs, seq, channels]
        core_out = core_normed_heads.transpose(1, 2).reshape(batch_size, -1, self.value_dim)
        
        attn_delta = self.linear_attn.out_proj(core_out.to(input_dtype))
        hidden_states = residual + attn_delta
        
        h_post = self.post_attention_layernorm(hidden_states)
        if self._use_moe:
            mlp_out = self.mlp(h_post)
        else:
            if fusion_level >= 2: # Fused GateUp
                gu = _qwen35_try_fused_awq_pair_matmul(
                    h_post,
                    self.mlp.gate_proj,
                    self.mlp.up_proj,
                    self,
                    "mlp_gate_up",
                )
                if gu is not None:
                    inter = self.config.intermediate_size
                    gate, up = gu.split(inter, dim=-1)
                    mlp_out = self.mlp.down_proj(F.silu(gate) * up)
                else:
                    mlp_out = self.mlp.down_proj(
                        F.silu(self.mlp.gate_proj(h_post)) * self.mlp.up_proj(h_post)
                    )
            else:
                mlp_out = self.mlp.down_proj(
                    F.silu(self.mlp.gate_proj(h_post)) * self.mlp.up_proj(h_post)
                )
        return hidden_states + mlp_out

class Qwen3_5FullAttentionLayer(nn.Module):
    """
    Full attention matching HF `Qwen3_5Attention`: fused q/gate from q_proj, per-head q_norm/k_norm,
    sigmoid gate on attention output before o_proj.
    """

    def __init__(self, config: LiteConfig, quant_config, prefix="", layer_idx: int = 0, use_moe: bool = False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self._use_moe = use_moe
        self.hidden_act = getattr(config, "hidden_act", "silu")
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
        if use_moe:
            self.mlp = Qwen3_5MoeSparseMoeBlock(config, quant_config, prefix=f"{prefix}.mlp")
        else:
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
        rope_params = getattr(config, "rope_parameters", {})
        mrope_section = rope_params.get("mrope_section")
        rotary_dim = int(config.head_dim * getattr(config, "partial_rotary_factor", 1.0))
        if mrope_section:
            self.rotary_emb = MRotaryEmbedding(
                config.head_dim,
                rotary_dim,
                config.max_position_embeddings,
                rope_params.get("rope_theta", 10000.0),
                is_neox_style=True,
                dtype=torch.float16,
                mrope_section=mrope_section,
                mrope_interleaved=rope_params.get("mrope_interleaved", True),
            )
        else:
            from .llama import get_rotary_embedding
            self.rotary_emb = get_rotary_embedding(config)
        # Legacy ROCm-friendly caps on full-attn matmuls / residuals (off by default for HF parity).
        self._use_full_attn_stabilizer = _env_truthy("FASTINFERENCE_QWEN35_FULLATTN_STABILIZER")
        self.kv_cache_dtype = os.environ.get("FASTINFERENCE_KV_TYPE", "auto")
        self._k_scale_float = float(os.environ.get("FASTINFERENCE_K_SCALE", "1.0"))
        self._v_scale_float = float(os.environ.get("FASTINFERENCE_V_SCALE", "1.0"))

    def forward(self, x, positions, kv_cache, attn_metadata):
        inf_config = attn_metadata.get("config") if isinstance(attn_metadata, dict) else getattr(attn_metadata, "config", None)
        fusion_level = inf_config.fusion_level if inf_config else 2
        
        input_dtype = x.dtype
        h = self.input_layernorm(x)
        bs, seq, _ = h.shape
        nh, nkv, hd = self.num_heads, self.num_kv_heads, self.head_dim
        n_tokens = bs * seq

        q_merged = self.self_attn.q_proj(h)
        # HF `Qwen3_5Attention`: view (..., num_heads, 2*head_dim) then `chunk(2, dim=-1)` — q and gate are
        # interleaved per head, NOT `[concat all Q heads | concat all gates]` (a plain split on dim=-1 is wrong).
        q_part, gate = torch.chunk(q_merged.view(bs, seq, nh, hd * 2), 2, dim=-1)

        # Per-head q_norm (weight dim=head_dim), same as HF.
        q_part = self.self_attn.q_norm(q_part.reshape(bs, -1, nh, hd)).reshape(bs, -1, nh, hd)

        kv_out = nkv * hd
        if fusion_level >= 2: # Fused self_attn_kv
            kv_merged = _qwen35_try_fused_awq_pair_matmul(
                h,
                self.self_attn.k_proj,
                self.self_attn.v_proj,
                self,
                "self_attn_kv",
            )
            if kv_merged is not None:
                k_raw, v_raw = kv_merged.split(kv_out, dim=-1)
                k = k_raw.view(bs, -1, nkv, hd)
                k = self.self_attn.k_norm(k).reshape(bs, -1, nkv, hd)
                v = v_raw.view(bs, -1, nkv, hd)
            else:
                k = self.self_attn.k_proj(h).view(bs, -1, nkv, hd)
                k = self.self_attn.k_norm(k).reshape(bs, -1, nkv, hd)
                v = self.self_attn.v_proj(h).view(bs, -1, nkv, hd)
        else:
            # K-Norm is often PER-HEAD in GGUF (256 weight)
            k = self.self_attn.k_proj(h).view(bs, -1, nkv, hd)
            k = self.self_attn.k_norm(k).reshape(bs, -1, nkv, hd)
            v = self.self_attn.v_proj(h).view(bs, -1, nkv, hd)

        q = q_part.reshape(n_tokens, nh, hd)
        k = k.reshape(n_tokens, nkv, hd)
        v = v.reshape(n_tokens, nkv, hd)
        q, k = self.rotary_emb(positions, q.unsqueeze(0), k.unsqueeze(0))
        q = q.squeeze(0)
        k = k.squeeze(0)
        k_cache, v_cache = kv_cache
        kv_scale_cache = attn_metadata.get("kv_scale_cache")
        if kv_scale_cache is not None:
            k_scale_cache, v_scale_cache = kv_scale_cache[self.layer_idx]
        else:
            k_scale_cache, v_scale_cache = (None, None)

        from vllm.kernels.triton.reshape_and_cache import reshape_and_cache
        from vllm.kernels.triton.paged_attention import paged_attention_v1
        
        kv_cache_dtype = inf_config.kv_type if inf_config else attn_metadata.get("kv_cache_dtype", self.kv_cache_dtype)
        k_scale = inf_config.k_scale if inf_config else attn_metadata.get("k_scale", self._k_scale_float)
        v_scale = inf_config.v_scale if inf_config else attn_metadata.get("v_scale", self._v_scale_float)
        
        reshape_and_cache(k, v, k_cache, v_cache, attn_metadata["slot_mapping"], kv_cache_dtype, k_scale, v_scale,
                          k_scale_cache=k_scale_cache, v_scale_cache=v_scale_cache)
        block_tables = attn_metadata["block_tables"]
        seq_lens = attn_metadata["seq_lens"]
        is_prefill = attn_metadata.get("is_prefill", False)
        max_ctx = max(nh * hd, getattr(self.config, "max_position_embeddings", 4096))
        # Chunked prefill: only the first chunk starts at token 0. Direct prefill ignores KV prefix;
        # use paged attention when this step continues after prior chunks (kv_start_indices > 0).
        kv_start_t = attn_metadata.get("kv_start_indices")
        kv_chunk_start = int(kv_start_t.reshape(-1)[0].item()) if kv_start_t is not None else 0

        # TurboQuant INT4 requires paged_attention for dequantization; skip direct prefill path
        is_int4 = "int4" in str(kv_cache_dtype).lower()
        use_direct_prefill = seq > 1 and is_prefill and kv_chunk_start == 0 and not is_int4
        use_sdpa_prefill = use_direct_prefill and _env_qwen35_sdpa_prefill_enabled()
        if use_direct_prefill and not use_sdpa_prefill:
            # HF-parity: same as transformers eager_attention_forward + causal (not Triton paged softmax).
            qh = q.view(bs, seq, nh, hd).transpose(1, 2)
            kh = k.view(bs, seq, nkv, hd).transpose(1, 2)
            vh = v.view(bs, seq, nkv, hd).transpose(1, 2)
            attn_b = _hf_eager_causal_attention_prefill(qh, kh, vh, hd**-0.5)
            attn_in = attn_b.reshape(n_tokens, nh, hd).to(dtype=q.dtype).contiguous()
        elif use_sdpa_prefill:
            qh = q.view(bs, seq, nh, hd).transpose(1, 2)
            kh = k.view(bs, seq, nkv, hd).transpose(1, 2)
            vh = v.view(bs, seq, nkv, hd).transpose(1, 2)
            if nh != nkv:
                kh = kh.repeat_interleave(nh // nkv, dim=1)
                vh = vh.repeat_interleave(nh // nkv, dim=1)
            attn_b = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True)
            attn_in = attn_b.transpose(1, 2).reshape(n_tokens, nh, hd).to(dtype=q.dtype).contiguous()
        else:
            attn_in = torch.empty((n_tokens, nh, hd), device=q.device, dtype=q.dtype)
            from vllm.engine.lite_engine import expand_metadata_for_paged_attention
            seq_lens_ext, block_tables_ext = expand_metadata_for_paged_attention(
                bs, seq, is_prefill, seq_lens, block_tables, q.device
            )

            # Kernel expects (batch_size, max_blocks) with stride_bt_seq.
            # For flattened prefill (n_tokens, max_blocks), we MUST use stride 0
            # for the logical "batch" dimension if we want it to treat each token
            # as a separate sequence correctly.
            class MockTensor:
                def __init__(self, t):
                    self.t = t
                    self.shape = t.shape
                def stride(self, dim=None):
                    if dim is None: return self.t.stride()
                    if dim == 0: return 0
                    return self.t.stride(dim)
                def __getattr__(self, name):
                    return getattr(self.t, name)

            paged_attention_v1(
                attn_in,
                q.contiguous(),
                k_cache,
                v_cache,
                nh,
                hd**-0.5,
                MockTensor(block_tables_ext),
                seq_lens_ext,
                k_cache.shape[1],
                max_ctx,
                None,
                kv_cache_dtype,
                k_scale,
                v_scale,
                k_scale_ptrs=k_scale_cache,
                v_scale_ptrs=v_scale_cache,
                num_kv_heads=nkv,
            )

        attn_flat = attn_in.reshape(bs, seq, nh * hd)
        gate = gate.reshape(bs, seq, nh * hd)
        attn_gated = attn_flat * torch.sigmoid(gate.to(dtype=attn_flat.dtype))

        if self._use_full_attn_stabilizer:
            attn_out = _call_litelinear_stable(self.self_attn.o_proj, attn_gated.float(), pre_cap=1024.0)
            hidden_states = _residual_merge_fp16(x.float(), attn_out, input_dtype, 8.0)
        else:
            attn_out = self.self_attn.o_proj(attn_gated)
            hidden_states = x + attn_out

        h_post = self.post_attention_layernorm(hidden_states)
        if self._use_moe:
            mlp_out = self.mlp(h_post)
            return hidden_states + mlp_out
        if self._use_full_attn_stabilizer:
            gp = self.mlp.gate_proj(h_post).float()
            up = self.mlp.up_proj(h_post).float()
            mlp_mid = F.silu(gp) * up
            mlp_out = _call_litelinear_stable(self.mlp.down_proj, mlp_mid, pre_cap=1536.0)
            return _residual_merge_fp16(hidden_states.float(), mlp_out, input_dtype, 8.0)

        if fusion_level >= 2: # Fused GateUp
            gu = _qwen35_try_fused_awq_pair_matmul(
                h_post,
                self.mlp.gate_proj,
                self.mlp.up_proj,
                self,
                "mlp_gate_up",
            )
            if gu is not None:
                inter = self.config.intermediate_size
                gate, up = gu.split(inter, dim=-1)
                mlp_out = self.mlp.down_proj(F.silu(gate) * up)
            else:
                mlp_out = self.mlp.down_proj(
                    F.silu(self.mlp.gate_proj(h_post)) * self.mlp.up_proj(h_post)
                )
        else:
            mlp_out = self.mlp.down_proj(
                F.silu(self.mlp.gate_proj(h_post)) * self.mlp.up_proj(h_post)
            )
        return hidden_states + mlp_out

    def load_weights(self, weights_iterator):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights_iterator:
            if "attn_norm.weight" in name:
                params_dict["input_layernorm.weight"].data.copy_(loaded_weight)
            elif "attn_q.weight" in name:
                params_dict["self_attn.q_proj.weight"].data.copy_(loaded_weight)
            elif "attn_q.bias" in name:
                params_dict["self_attn.q_proj.bias"].data.copy_(loaded_weight)
            elif "attn_k.weight" in name:
                params_dict["self_attn.k_proj.weight"].data.copy_(loaded_weight)
            elif "attn_k.bias" in name:
                params_dict["self_attn.k_proj.bias"].data.copy_(loaded_weight)
            elif "attn_v.weight" in name:
                params_dict["self_attn.v_proj.weight"].data.copy_(loaded_weight)
            elif "attn_v.bias" in name:
                params_dict["self_attn.v_proj.bias"].data.copy_(loaded_weight)
            elif "attn_output.weight" in name:
                params_dict["self_attn.o_proj.weight"].data.copy_(loaded_weight)
            elif "attn_q_norm.weight" in name:
                params_dict["self_attn.q_norm.weight"].data.copy_(loaded_weight)
            elif "attn_k_norm.weight" in name:
                params_dict["self_attn.k_norm.weight"].data.copy_(loaded_weight)
            elif "post_attention_norm.weight" in name:
                params_dict["post_attention_layernorm.weight"].data.copy_(loaded_weight)
            elif "ffn_gate.weight" in name:
                params_dict["mlp.gate_proj.weight"].data.copy_(loaded_weight)
            elif "ffn_up.weight" in name:
                params_dict["mlp.up_proj.weight"].data.copy_(loaded_weight)
            elif "ffn_down.weight" in name:
                params_dict["mlp.down_proj.weight"].data.copy_(loaded_weight)

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
                        self.config, quant_config, f"model.layers.{i}", layer_idx=i
                    )
                )
        self.norm = Qwen3_5RMSNorm(self.config.hidden_size, eps=getattr(self.config, "rms_norm_eps", 1e-6))


class Qwen2MoEModel(nn.Module):
    """Qwen3.5 MoE text: same attention stack as ``Qwen2Model`` but sparse MoE FFN (HF-aligned)."""

    def __init__(self, hf_config, quant_config):
        super().__init__()
        self.config = LiteConfig(hf_config)
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.layers = nn.ModuleList()
        for i in range(self.config.num_hidden_layers):
            if qwen35_layer_type(i, self.config) == "linear_attention":
                self.layers.append(
                    Qwen3_5LinearAttentionLayer(
                        self.config, quant_config, f"model.layers.{i}", layer_idx=i, use_moe=True
                    )
                )
            else:
                self.layers.append(
                    Qwen3_5FullAttentionLayer(self.config, quant_config, f"model.layers.{i}", layer_idx=i, use_moe=True)
                )
        self.norm = Qwen3_5RMSNorm(self.config.hidden_size, eps=getattr(self.config, "rms_norm_eps", 1e-6))

class Qwen3_5ForConditionalGeneration(nn.Module):
    def __init__(self, vllm_config, prefix=""):
        super().__init__()
        self.model = Qwen2Model(vllm_config.model_config.hf_config, vllm_config.quant_config)
        self.lm_head = nn.Linear(self.model.config.hidden_size, self.model.config.vocab_size, bias=False)
    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        # Match Hugging Face Qwen3_5TextModel: MRoPE uses (3, batch, seq) position_ids (text uses
        # three identical rows after the 4-way split in HF). LiteEngine supplies 2D positions.
        rp = getattr(self.model.config, "rope_parameters", None) or {}
        if positions is not None and positions.ndim == 2 and rp.get("mrope_section"):
            positions = positions.unsqueeze(0).expand(3, -1, -1).contiguous()
        x = self.model.embed_tokens(input_ids)
        for i in range(len(self.model.layers)):
            x = self.model.layers[i](x, positions, kv_caches[i], attn_metadata)
        return self.lm_head(self.model.norm(x))

class Qwen3_5MoeForConditionalGeneration(nn.Module):
    def __init__(self, vllm_config, prefix=""):
        super().__init__()
        self.model = Qwen2MoEModel(vllm_config.model_config.hf_config, vllm_config.quant_config)
        self.lm_head = nn.Linear(self.model.config.hidden_size, self.model.config.vocab_size, bias=False)

    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        rp = getattr(self.model.config, "rope_parameters", None) or {}
        if positions is not None and positions.ndim == 2 and rp.get("mrope_section"):
            positions = positions.unsqueeze(0).expand(3, -1, -1).contiguous()
        x = self.model.embed_tokens(input_ids)
        for i in range(len(self.model.layers)):
            x = self.model.layers[i](x, positions, kv_caches[i], attn_metadata)
        return self.lm_head(self.model.norm(x))
