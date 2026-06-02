# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe
from vllm.model_executor.layers.quantization.tensor import (
    dequantize_awq_pytorch,
    dequantize_symmetric_packed_int4_pytorch,
)

from .config import Gemma4LayerConfig, _GEMMA4_MOE_MATERIALIZE_BATCH_EXPERTS
from vllm.model_executor.models.lite_config import LiteConfig
from .mlp import Gemma4MLP
from .policy_utils import (
    _gemma4_policy_value,
    _get_eps,
    _reshape_hidden_to_2d,
    _restore_hidden_from_2d,
)
from .profiling import _gemma4_profile_span


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


def _resolve_gemma4_moe_compute_dtype(
    policy: str,
    hidden_dtype: torch.dtype,
) -> torch.dtype:
    normalized = str(policy or "auto").strip().lower()
    if normalized in ("fp32", "float32"):
        return torch.float32
    if normalized in ("fp16", "float16"):
        return torch.float16
    if normalized in ("bf16", "bfloat16"):
        return torch.bfloat16
    return hidden_dtype


class Gemma4TopKRouterLite(nn.Module):
    def __init__(self, config: LiteConfig, quant_config: Any, prefix: str):
        super().__init__()
        self.num_experts = int(config.num_experts)
        self.top_k = int(config.num_experts_per_tok)
        self.hidden_size = int(config.hidden_size)
        self.eps = float(_get_eps(config))
        self.scalar_root_size = float(max(1, self.hidden_size)) ** -0.5
        from vllm.model_executor.layers.lite_linear import LiteLinear

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
    layer: Any,
    *,
    out_features: int,
    in_features: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    from vllm.model_executor.layers.lite_linear import LiteLinear

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
            f"Layer '{getattr(layer, 'prefix', '<unknown>')}' has neither dense "
            "nor packed weights."
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
            f"Layer '{getattr(layer, 'prefix', '<unknown>')}' dequantized weight "
            "too small: "
            f"got {tuple(dense_weight.shape)}, need ({out_features}, {in_features})"
        )
    return (
        dense_weight[:out_features, :in_features]
        .contiguous()
        .to(device=device, dtype=dtype)
    )


class Gemma4MoeExpertsLite(nn.Module):
    def __init__(
        self,
        config: LiteConfig,
        quant_config: Any,
        prefix: str,
        runtime_config: Any = None,
    ):
        super().__init__()
        self.hidden_dim = int(config.hidden_size)
        self.intermediate_dim = int(config.moe_intermediate_size)
        self.num_experts = int(config.num_experts)
        self.top_k = int(config.num_experts_per_tok)
        self.hidden_act = str(
            getattr(config, "hidden_activation", getattr(config, "hidden_act", "silu"))
        ).lower()
        from vllm.model_executor.layers.lite_linear import LiteLinear

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
        self._cached_device: torch.device | None = None
        self._cached_dtype: torch.dtype | None = None
        self._cached_w1: torch.Tensor | None = None
        self._cached_w2: torch.Tensor | None = None
        self._expert_weight_cache: OrderedDict[
            int, tuple[torch.Tensor, torch.Tensor]
        ] = OrderedDict()
        self._expert_cache_device: torch.device | None = None
        self._expert_cache_dtype: torch.dtype | None = None
        self._max_expert_cache = max(
            0,
            int(
                _gemma4_policy_value(
                    runtime_config,
                    "moe_expert_cache_size",
                    getattr(runtime_config, "gemma4_moe_expert_cache_size", 32),
                )
            ),
        )
        self._compute_dtype_policy = str(
            _gemma4_policy_value(
                runtime_config,
                "moe_compute_dtype",
                getattr(runtime_config, "gemma4_moe_compute_dtype", "auto"),
            )
        )
        self._int4_kernel_enabled = bool(
            _gemma4_policy_value(
                runtime_config,
                "moe_int4_kernel_enabled",
                getattr(runtime_config, "gemma4_moe_int4_kernel_enabled", True),
            )
        )
        self._int4_kernel_strategy = (
            str(
                _gemma4_policy_value(
                    runtime_config,
                    "moe_int4_kernel_strategy",
                    getattr(
                        runtime_config,
                        "gemma4_moe_int4_kernel_strategy",
                        "two_stage",
                    ),
                )
            )
            .strip()
            .lower()
        )
        self._prefill_grouped_enabled = bool(
            _gemma4_policy_value(
                runtime_config,
                "moe_prefill_grouped_enabled",
                getattr(runtime_config, "gemma4_moe_prefill_grouped_enabled", False),
            )
        )
        self._prefill_grouped_min_tokens = max(
            1,
            int(
                _gemma4_policy_value(
                    runtime_config,
                    "moe_prefill_grouped_min_tokens",
                    getattr(
                        runtime_config,
                        "gemma4_moe_prefill_grouped_min_tokens",
                        17,
                    ),
                )
            ),
        )
        self._prefill_grouped_strategy = (
            str(
                _gemma4_policy_value(
                    runtime_config,
                    "moe_prefill_grouped_strategy",
                    getattr(
                        runtime_config,
                        "gemma4_moe_prefill_grouped_strategy",
                        "chunked",
                    ),
                )
            )
            .strip()
            .lower()
        )
        if self._prefill_grouped_strategy not in ("chunked", "fused"):
            self._prefill_grouped_strategy = "chunked"
        self._batch_materialize_enabled = bool(
            _gemma4_policy_value(
                runtime_config,
                "moe_batch_materialize_enabled",
                getattr(runtime_config, "gemma4_moe_batch_materialize_enabled", False),
            )
        )
        self._layer_config = Gemma4LayerConfig()

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
        with _gemma4_profile_span("moe_materialize_one_expert_awq", self._layer_config):
            return self._materialize_one_expert_awq_impl(expert_id, device, dtype)

    def _materialize_one_expert_awq_impl(
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
        if self._expert_cache_device != device or self._expert_cache_dtype != dtype:
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
        w1 = (
            w1e[: 2 * self.intermediate_dim, : self.hidden_dim]
            .contiguous()
            .to(device=device, dtype=dtype)
        )
        w2 = (
            w2e[: self.hidden_dim, : self.intermediate_dim]
            .contiguous()
            .to(device=device, dtype=dtype)
        )
        if self._max_expert_cache > 0:
            self._expert_weight_cache[expert_id] = (w1, w2)
            self._expert_weight_cache.move_to_end(expert_id)
            while len(self._expert_weight_cache) > self._max_expert_cache:
                self._expert_weight_cache.popitem(last=False)
        return w1, w2

    def _materialize_experts_awq_batch(
        self,
        expert_ids: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        qweight_gu = self.gate_up_proj.qweight
        scales_gu = self.gate_up_proj.scales
        qweight_d = self.down_proj.qweight
        scales_d = self.down_proj.scales
        if (
            not isinstance(qweight_gu, torch.Tensor)
            or not isinstance(scales_gu, torch.Tensor)
            or qweight_gu.ndim != 3
            or scales_gu.ndim != 3
            or not isinstance(qweight_d, torch.Tensor)
            or not isinstance(scales_d, torch.Tensor)
            or qweight_d.ndim != 3
            or scales_d.ndim != 3
        ):
            return {}

        gsz_gu = max(1, int((qweight_gu.shape[2] * 8) // max(1, scales_gu.shape[2])))
        gsz_d = max(1, int((qweight_d.shape[2] * 8) // max(1, scales_d.shape[2])))
        materialized: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for offset in range(
            0,
            int(expert_ids.numel()),
            _GEMMA4_MOE_MATERIALIZE_BATCH_EXPERTS,
        ):
            ids = expert_ids[
                offset : offset + _GEMMA4_MOE_MATERIALIZE_BATCH_EXPERTS
            ].to(device=qweight_gu.device, dtype=torch.long)
            with _gemma4_profile_span(
                "moe_materialize_expert_batch_awq", self._layer_config
            ):
                qweight_gu_batch = qweight_gu.index_select(0, ids).to(
                    device=device,
                    dtype=torch.int32,
                )
                qweight_d_batch = qweight_d.index_select(0, ids).to(
                    device=device,
                    dtype=torch.int32,
                )
                w1_batch = dequantize_symmetric_packed_int4_pytorch(
                    qweight_gu_batch,
                    scales_gu.index_select(0, ids).to(device=device),
                    group_size=gsz_gu,
                )
                w2_batch = dequantize_symmetric_packed_int4_pytorch(
                    qweight_d_batch,
                    scales_d.index_select(0, ids).to(device=device),
                    group_size=gsz_d,
                )
            w1_batch = (
                w1_batch[:, : 2 * self.intermediate_dim, : self.hidden_dim]
                .contiguous()
                .to(device=device, dtype=dtype)
            )
            w2_batch = (
                w2_batch[:, : self.hidden_dim, : self.intermediate_dim]
                .contiguous()
                .to(device=device, dtype=dtype)
            )
            for batch_idx, expert_id in enumerate(ids.tolist()):
                materialized[int(expert_id)] = (
                    w1_batch[batch_idx],
                    w2_batch[batch_idx],
                )
        return materialized

    def _forward_awq_streaming(
        self,
        hidden_states_2d: torch.Tensor,
        router_logits: torch.Tensor | None,
        topk_weights: torch.Tensor | None = None,
        topk_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if topk_weights is None or topk_ids is None:
            if router_logits is None:
                raise RuntimeError(
                    "router_logits or top-k routing inputs are required."
                )
            topk_weights, topk_ids = torch.topk(
                router_logits,
                k=self.top_k,
                dim=-1,
            )
            topk_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32).to(
                hidden_states_2d.dtype
            )
        compute_dtype = _resolve_gemma4_moe_compute_dtype(
            self._compute_dtype_policy,
            hidden_states_2d.dtype,
        )
        n_tokens = int(hidden_states_2d.shape[0])
        if (
            self._prefill_grouped_enabled
            and n_tokens >= self._prefill_grouped_min_tokens
        ):
            grouped_out = self._forward_awq_grouped_prefill(
                hidden_states_2d,
                router_logits,
                topk_weights,
                topk_ids,
                compute_dtype,
            )
            if grouped_out is not None:
                return grouped_out.to(hidden_states_2d.dtype)
        if self._int4_kernel_enabled:
            from vllm.kernels.triton.gemma4_moe_int4 import (
                gemma4_moe_int4_decode,
                gemma4_moe_int4_decode_batched,
                gemma4_moe_int4_decode_batched_chunked,
                gemma4_moe_int4_decode_batched_chunked_downpair,
                gemma4_moe_int4_decode_batched_chunked_pair,
                gemma4_moe_int4_decode_batched_chunked_splitgate_downpair,
                gemma4_moe_int4_decode_batched_grouped,
                gemma4_moe_int4_decode_batched_grouped_streaming,
                gemma4_moe_int4_decode_batched_tuned,
                gemma4_moe_int4_decode_single_kernel,
            )

            decode_kernel = (
                gemma4_moe_int4_decode_single_kernel
                if self._int4_kernel_strategy == "single"
                else gemma4_moe_int4_decode_batched_tuned
                if self._int4_kernel_strategy == "batched_tuned"
                else gemma4_moe_int4_decode_batched_chunked_pair
                if self._int4_kernel_strategy == "batched_chunked_pair"
                else gemma4_moe_int4_decode_batched_chunked_downpair
                if self._int4_kernel_strategy == "batched_chunked_downpair"
                else gemma4_moe_int4_decode_batched_chunked_splitgate_downpair
                if self._int4_kernel_strategy == "batched_chunked_splitgate_downpair"
                else gemma4_moe_int4_decode_batched_grouped
                if self._int4_kernel_strategy == "batched_grouped"
                else gemma4_moe_int4_decode_batched_grouped_streaming
                if self._int4_kernel_strategy == "batched_grouped_streaming"
                else gemma4_moe_int4_decode_batched_chunked
                if self._int4_kernel_strategy == "batched_chunked"
                else gemma4_moe_int4_decode_batched
                if self._int4_kernel_strategy == "batched"
                else gemma4_moe_int4_decode
            )

            with _gemma4_profile_span("moe_int4_decode_attempt", self._layer_config):
                fast_out, used_fast, fast_reason = decode_kernel(
                    hidden_states_2d,
                    topk_weights,
                    topk_ids,
                    getattr(self.gate_up_proj, "qweight", torch.empty(0)),
                    getattr(self.gate_up_proj, "scales", torch.empty(0)),
                    getattr(self.down_proj, "qweight", torch.empty(0)),
                    getattr(self.down_proj, "scales", torch.empty(0)),
                    intermediate_dim=self.intermediate_dim,
                    activation=self.hidden_act,
                )
            if used_fast:
                if self._layer_config.profile_enabled:
                    bucket = self._layer_config.profile_stats.setdefault(
                        "moe_int4_decode_used", {"time_s": 0.0, "count": 0.0}
                    )
                    bucket["time_s"] += 0.0
                    bucket["count"] += 1.0
                return fast_out.to(hidden_states_2d.dtype)
            bucket = self._layer_config.profile_stats.setdefault(
                f"moe_int4_decode_fallback:{fast_reason}",
                {"time_s": 0.0, "count": 0.0},
            )
            bucket["time_s"] += 0.0
            bucket["count"] += 1.0
        out = torch.zeros_like(hidden_states_2d, dtype=compute_dtype)
        flat_topk_ids = topk_ids.reshape(-1).to(torch.long)
        flat_topk_weights = topk_weights.reshape(-1)
        flat_token_idx = torch.arange(
            n_tokens, device=hidden_states_2d.device, dtype=torch.long
        ).repeat_interleave(self.top_k)

        sorted_expert_ids, sort_idx = torch.sort(flat_topk_ids)
        sorted_token_idx = flat_token_idx.index_select(0, sort_idx)
        sorted_weights = flat_topk_weights.index_select(0, sort_idx)
        unique_experts, counts = torch.unique_consecutive(
            sorted_expert_ids, return_counts=True
        )
        batched_expert_weights: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        if (
            self._batch_materialize_enabled
            and n_tokens > 1
            and int(unique_experts.numel()) > 1
        ):
            batched_expert_weights = self._materialize_experts_awq_batch(
                unique_experts,
                hidden_states_2d.device,
                compute_dtype,
            )

        start = 0
        unique_experts_cpu = unique_experts.to(device="cpu").tolist()
        counts_cpu = counts.to(device="cpu").tolist()
        for expert_id, count in zip(unique_experts_cpu, counts_cpu):
            if count <= 0:
                continue
            end = start + count
            token_idx = sorted_token_idx[start:end]
            coeff = sorted_weights[start:end].unsqueeze(-1).to(compute_dtype)
            start = end
            x_sel = hidden_states_2d.index_select(0, token_idx).to(compute_dtype)
            expert_weights = batched_expert_weights.get(expert_id)
            if expert_weights is not None:
                w1e, w2e = expert_weights
            else:
                w1e, w2e = self._materialize_one_expert_awq(
                    expert_id,
                    hidden_states_2d.device,
                    compute_dtype,
                )
            with _gemma4_profile_span("moe_sparse_expert_linear", self._layer_config):
                with _gemma4_profile_span(
                    "moe_sparse_expert_gate_up", self._layer_config
                ):
                    gu = F.linear(x_sel, w1e)
                g, u = torch.chunk(gu, 2, dim=-1)
                h = self._apply_gate_activation(g) * u
                with _gemma4_profile_span(
                    "moe_sparse_expert_down_reduce", self._layer_config
                ):
                    y = F.linear(h, w2e) * coeff
            out.index_add_(0, token_idx, y)
        return out.to(hidden_states_2d.dtype)

    def _forward_awq_grouped_prefill(
        self,
        hidden_states_2d: torch.Tensor,
        router_logits: torch.Tensor | None,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        compute_dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if hidden_states_2d.ndim != 2 or int(hidden_states_2d.shape[0]) <= 1:
            return None
        del router_logits, compute_dtype
        from vllm.kernels.triton.gemma4_moe_int4 import (
            gemma4_moe_int4_prefill_grouped,
            gemma4_moe_int4_prefill_grouped_fused,
        )

        prefill_kernel = (
            gemma4_moe_int4_prefill_grouped_fused
            if self._prefill_grouped_strategy == "fused"
            else gemma4_moe_int4_prefill_grouped
        )

        with _gemma4_profile_span("moe_awq_grouped_prefill", self._layer_config):
            out, used, _ = prefill_kernel(
                hidden_states_2d,
                topk_weights,
                topk_ids,
                getattr(self.gate_up_proj, "qweight", torch.empty(0)),
                getattr(self.gate_up_proj, "scales", torch.empty(0)),
                getattr(self.down_proj, "qweight", torch.empty(0)),
                getattr(self.down_proj, "scales", torch.empty(0)),
                intermediate_dim=self.intermediate_dim,
                activation=self.hidden_act,
            )
        return out if used else None

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
            gsz_gu = max(
                1, int((qweight_gu.shape[2] * 8) // max(1, scales_gu.shape[2]))
            )
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
                    w1e[: 2 * self.intermediate_dim, : self.hidden_dim].to(
                        device=device, dtype=dtype
                    )
                )
                w2_parts.append(
                    w2e[: self.hidden_dim, : self.intermediate_dim].to(
                        device=device, dtype=dtype
                    )
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
        router_logits: torch.Tensor | None = None,
        topk_weights: torch.Tensor | None = None,
        topk_ids: torch.Tensor | None = None,
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
            raise RuntimeError(
                "router_logits is required when top-k routing is not provided."
            )
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
    def __init__(
        self,
        config: LiteConfig,
        quant_config: Any,
        prefix: str,
        runtime_config: Any = None,
    ):
        super().__init__()
        self.router = Gemma4TopKRouterLite(config, quant_config, prefix)
        self.experts = Gemma4MoeExpertsLite(
            config, quant_config, prefix, runtime_config=runtime_config
        )
        self.shared_mlp = Gemma4MLP(config, quant_config, prefix)
        self._layer_config = Gemma4LayerConfig()

    def forward_branches(
        self,
        hidden_states_dense: torch.Tensor,
        hidden_states_sparse: torch.Tensor,
        hidden_states_router: torch.Tensor | None = None,
        lora_mapping: Any = None,
        inf_config: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states_2d, shape = _reshape_hidden_to_2d(hidden_states_sparse)
        router_src = (
            hidden_states_router
            if hidden_states_router is not None
            else hidden_states_sparse
        )
        router_2d, _ = _reshape_hidden_to_2d(router_src)
        router_logits, routing_weights, selected_experts = self.router(router_2d)
        sparse_out_2d = self.experts(
            hidden_states_2d,
            router_logits,
            topk_weights=routing_weights,
            topk_ids=selected_experts,
        )
        sparse_out = _restore_hidden_from_2d(sparse_out_2d, shape)
        dense_out = self.shared_mlp(
            hidden_states_dense, lora_mapping, inf_config=inf_config
        )
        return dense_out, sparse_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        lora_mapping: Any = None,
        inf_config: Any = None,
    ) -> torch.Tensor:
        dense_out, sparse_out = self.forward_branches(
            hidden_states,
            hidden_states,
            hidden_states_router=hidden_states,
            lora_mapping=lora_mapping,
            inf_config=inf_config,
        )
        return dense_out + sparse_out
