# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.layers.quantization.tensor import record_awq_audit_event
from vllm.model_executor.models.lite_config import LiteConfig

from .config import Gemma4LayerConfig
from .kv_utils import (
    _causal_attention_ref,
    _gather_recent_kv,
    _gather_recent_kv_batched,
    _get_or_build_local_decode_aligned_metadata,
    _is_packed_or_quantized_kv_cache,
    _local_prefill_attention_sdpa,
    _should_use_full_decode_reference,
    _use_legacy_full_precision_kv_write,
    _write_full_precision_kv_cache,
)
from .policy_utils import (
    _gemma4_model_policy_truthy,
    _get_eps,
    _meta_cpu_max_seq_len,
    _meta_cpu_seq_lens,
    _meta_get,
    _resolve_max_position_plus_one_cpu,
)
from .profiling import _gemma4_profile_span
from .rope import _get_rope_with_runtime, _is_local_layer, _layer_type_for_idx


def _linear_quant_attr(layer: LiteLinear, name: str) -> torch.Tensor | None:
    value = getattr(layer, name, None)
    return value if isinstance(value, torch.Tensor) and value.numel() > 1 else None


def _try_fused_awq_qkv_decode(
    x: torch.Tensor,
    q_proj: LiteLinear,
    k_proj: LiteLinear,
    v_proj: LiteLinear | None,
    *,
    inf_config: Any = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    x2 = x.reshape(-1, x.shape[-1])
    if int(x2.shape[0]) != 1:
        return None
    q_qweight = _linear_quant_attr(q_proj, "qweight")
    k_qweight = _linear_quant_attr(k_proj, "qweight")
    q_scales = _linear_quant_attr(q_proj, "scales")
    k_scales = _linear_quant_attr(k_proj, "scales")
    if q_qweight is None or k_qweight is None:
        return None
    if q_scales is None or k_scales is None:
        return None
    qzeros = _linear_quant_attr(q_proj, "qzeros")
    kzeros = _linear_quant_attr(k_proj, "qzeros")
    if qzeros is not None or kzeros is not None:
        return None
    group_size = int(getattr(q_proj, "group_size", 128))
    if int(getattr(k_proj, "group_size", 128)) != group_size:
        return None

    v_qweight = None
    v_scales = None
    if v_proj is not None:
        v_qweight = _linear_quant_attr(v_proj, "qweight")
        v_scales = _linear_quant_attr(v_proj, "scales")
        vzeros = _linear_quant_attr(v_proj, "qzeros")
        if v_qweight is None or v_scales is None or vzeros is not None:
            return None
        if int(getattr(v_proj, "group_size", 128)) != group_size:
            return None

    try:
        from vllm.kernels.triton.awq_fused_gemm import (
            packed_int4_symmetric_fused_qkv_m1_safe,
        )

        fused, used, _ = packed_int4_symmetric_fused_qkv_m1_safe(
            x2.contiguous(),
            q_qweight,
            k_qweight,
            v_qweight,
            q_scales,
            k_scales,
            v_scales,
            group_size,
            config=inf_config,
        )
    except Exception:
        return None
    if not used:
        return None

    q_n = int(q_qweight.shape[0])
    k_n = int(k_qweight.shape[0])
    q = fused[:, :q_n]
    k = fused[:, q_n : q_n + k_n]
    v = k if v_proj is None else fused[:, q_n + k_n :]
    lead_shape = x.shape[:-1]
    return (
        q.reshape(*lead_shape, q_n),
        k.reshape(*lead_shape, k_n),
        v.reshape(*lead_shape, int(v.shape[-1])),
    )


class Gemma4Attention(nn.Module):
    def __init__(
        self,
        config: LiteConfig,
        quant_config: Any,
        prefix: str,
        layer_idx: int,
        runtime_config: Any = None,
        kv_shared_with: Gemma4Attention | None = None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.kv_scale_cache_idx = layer_idx
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
            else getattr(
                config, "num_global_key_value_heads", config.num_key_value_heads
            )
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
        self._layer_config = Gemma4LayerConfig()
        self.rotary_emb = _get_rope_with_runtime(
            config,
            self.head_dim,
            self.layer_type,
            runtime_config=runtime_config,
            layer_config=self._layer_config,
        )
        self.is_kv_shared_layer = kv_shared_with is not None
        if self.is_kv_shared_layer:
            # KV-shared layers do not compute their own keys/values; they reuse
            # the donor layer's KV cache state (same kv_scale_cache_idx).  Leave
            # k/v projection and normalization modules undefined so the sharing
            # is explicit and we cannot accidentally overwrite the donor state.
            self.kv_scale_cache_idx = kv_shared_with.kv_scale_cache_idx
            self.k_proj = None
            self.v_proj = None
            self.k_norm = None

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
        inf_config = _meta_get(attn_metadata, "config", None)
        is_prefill_for_audit = bool(_meta_get(attn_metadata, "is_prefill", False))
        is_decode_m1 = int(x.reshape(-1, x.shape[-1]).shape[0]) == 1
        attn_prefix = getattr(
            self.q_proj,
            "prefix",
            "<unknown>.self_attn.q_proj",
        ).rsplit(".", 1)[0]
        fused_qkv = None
        if self.is_kv_shared_layer:
            # KV-shared layers reuse the donor layer's KV cache state.  Only the
            # query is computed here; the keys/values are read from the donor's
            # cache slot during attention.
            with _gemma4_profile_span("attn_q_proj", self._layer_config):
                q = self.q_proj(x, lora_mapping, inf_config=inf_config)
        else:
            if (not is_prefill_for_audit) and is_decode_m1:
                fused_qkv = _try_fused_awq_qkv_decode(
                    x,
                    self.q_proj,
                    self.k_proj,
                    self.v_proj,
                    inf_config=inf_config,
                )
            if fused_qkv is not None:
                q, k, v = fused_qkv
                event = "qk_fused_decode" if self.v_proj is None else "qkv_fused_decode"
                record_awq_audit_event(
                    attn_prefix,
                    event,
                    shape={
                        "m": 1,
                        "hidden": int(x.shape[-1]),
                        "q": int(self.q_size),
                        "k": int(self.kv_size),
                        "v": 0 if self.v_proj is None else int(self.kv_size),
                    },
                    reason="packed_int4_symmetric_fused_qkv_m1_safe",
                )
            else:
                if (not is_prefill_for_audit) and is_decode_m1:
                    event = (
                        "qk_separate_decode"
                        if self.v_proj is None
                        else "qkv_separate_decode"
                    )
                    shape = {
                        "m": 1,
                        "hidden": int(x.shape[-1]),
                        "q": int(self.q_size),
                        "k": int(self.kv_size),
                    }
                    if self.v_proj is not None:
                        shape["v"] = int(self.kv_size)
                    record_awq_audit_event(
                        attn_prefix,
                        event,
                        shape=shape,
                        reason="gemma4_attention_forward_uses_separate_litelinears",
                    )
                with _gemma4_profile_span("attn_q_proj", self._layer_config):
                    q = self.q_proj(x, lora_mapping, inf_config=inf_config)
                with _gemma4_profile_span("attn_k_proj", self._layer_config):
                    k = self.k_proj(x, lora_mapping, inf_config=inf_config)
                if self.v_proj is not None:
                    with _gemma4_profile_span("attn_v_proj", self._layer_config):
                        v = self.v_proj(x, lora_mapping, inf_config=inf_config)
                else:
                    v = k
        bsz, seqlen = x.shape[:2]
        q = q.view(bsz, seqlen, self.num_heads, self.head_dim)
        q = self._apply_head_norm(self.q_norm, q)
        if not self.is_kv_shared_layer:
            k = k.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
            v = v.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
            k = self._apply_head_norm(self.k_norm, k)
            v = self._apply_head_norm_noscale(v, self.v_norm_eps)
        # Surface the max position upper bound from the engine-side builder so
        # rotary_emb can extend its cache without a per-layer D->H sync.
        max_pos_plus_one_cpu = _resolve_max_position_plus_one_cpu(
            attn_metadata, positions
        )
        if self.is_kv_shared_layer:
            # Only the query needs rotary encoding; the donor keys were already
            # rotated when they were written to the KV cache.
            q, _ = self.rotary_emb(
                positions,
                q,
                q,
                max_position_plus_one_cpu=max_pos_plus_one_cpu,
                inf_config=inf_config,
            )
        else:
            q, k = self.rotary_emb(
                positions,
                q,
                k,
                max_position_plus_one_cpu=max_pos_plus_one_cpu,
                inf_config=inf_config,
            )
        is_local = self.is_sliding
        local_window = (
            int(getattr(self.config, "sliding_window", 0) or 0) if is_local else None
        )
        softcap = getattr(self.config, "attn_logit_softcapping", None)
        slot_mapping = _meta_get(attn_metadata, "slot_mapping", None)
        if slot_mapping is not None and kv_cache is not None:
            from vllm.kernels.triton.reshape_and_cache import reshape_and_cache

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
                k_scale_cache, v_scale_cache = kv_scale_cache[self.kv_scale_cache_idx]
            else:
                k_scale_cache, v_scale_cache = (None, None)

            if not self.is_kv_shared_layer:
                if (
                    _use_legacy_full_precision_kv_write(inf_config)
                    and _should_use_full_decode_reference(
                        inf_config, str(kv_cache_dtype)
                    )
                    and (not _is_packed_or_quantized_kv_cache(str(kv_cache_dtype)))
                ):
                    with _gemma4_profile_span(
                        "kv_write_full_precision", self._layer_config
                    ):
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
                    with _gemma4_profile_span(
                        "kv_write_reshape_and_cache", self._layer_config
                    ):
                        reshape_and_cache(
                            k.reshape(
                                -1, self.num_kv_heads, self.head_dim
                            ).contiguous(),
                            v.reshape(
                                -1, self.num_kv_heads, self.head_dim
                            ).contiguous(),
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
                with _gemma4_profile_span("attn_local_prefill", self._layer_config):
                    block_tables = _meta_get(attn_metadata, "block_tables", None)
                    seq_lens = _meta_get(attn_metadata, "seq_lens", None)
                    kv_start_t = _meta_get(attn_metadata, "kv_start_indices", None)
                    if kv_start_t is None:
                        q_starts = (
                            seq_lens.to(device=q.device, dtype=torch.long) - seqlen
                        )
                    else:
                        q_starts = kv_start_t.to(
                            device=q.device, dtype=torch.long
                        ).reshape(-1)
                    q_positions = (
                        q_starts[:, None]
                        + torch.arange(seqlen, device=q.device, dtype=torch.long)[
                            None, :
                        ]
                    )
                    with _gemma4_profile_span(
                        "kv_read_local_prefill", self._layer_config
                    ):
                        k_ctx, v_ctx, k_positions, k_valid = _gather_recent_kv_batched(
                            kv_cache=kv_cache,
                            block_tables=block_tables,
                            seq_lens=seq_lens,
                            num_kv_heads=self.num_kv_heads,
                            head_dim=self.head_dim,
                            local_window=int(local_window or 0),
                            kv_cache_dtype=str(kv_cache_dtype),
                            inf_config=inf_config,
                            kv_scale_cache=(k_scale_cache, v_scale_cache),
                            seq_lens_cpu=_meta_cpu_seq_lens(attn_metadata),
                            max_seq_len_cpu=_meta_cpu_max_seq_len(attn_metadata),
                        )
                    if softcap is not None and float(softcap) > 0:
                        out = (
                            _causal_attention_ref(
                                q.transpose(1, 2).float(),
                                k_ctx.transpose(1, 2).float(),
                                v_ctx.transpose(1, 2).float(),
                                self.scale,
                                local_window=None,
                                softcap=softcap,
                                key_padding_mask=k_valid,
                                q_positions=q_positions,
                                k_positions=k_positions,
                            )
                            .to(q.dtype)
                            .view(bsz, seqlen, -1)
                        )
                    else:
                        out = (
                            _local_prefill_attention_sdpa(
                                q,
                                k_ctx,
                                v_ctx,
                                q_positions,
                                k_positions,
                                k_valid,
                                local_window=None,
                                scale=self.scale,
                            )
                            .to(q.dtype)
                            .view(bsz, seqlen, -1)
                        )
            elif is_local and not is_prefill:
                with _gemma4_profile_span("attn_local_decode", self._layer_config):
                    block_tables = _meta_get(attn_metadata, "block_tables", None)
                    seq_lens = _meta_get(attn_metadata, "seq_lens", None)
                    kv_dtype_name = (
                        inf_config.kv_type
                        if inf_config is not None
                        else _meta_get(attn_metadata, "kv_cache_dtype", "auto")
                    )
                    ctx_window = int(local_window or 0)
                    # Step 3 rewire: the Triton paged-attention kernel now
                    # applies Gemma-style softcap internally (scale -> softcap
                    # -> -inf mask -> online softmax), so we no longer need to
                    # route softcap>0 through the eager pytorch ref path here.
                    use_triton_local_decode = (
                        _gemma4_model_policy_truthy(
                            inf_config, "local_decode_triton", default=True
                        )
                        and block_tables is not None
                        and seq_lens is not None
                        and seqlen == 1
                    )
                    if use_triton_local_decode:
                        from vllm.kernels.triton.paged_attention import (
                            paged_attention_v1,
                        )

                        attn_out = torch.empty(
                            (bsz * seqlen, self.num_heads, self.head_dim),
                            device=q.device,
                            dtype=q.dtype,
                        )
                        with _gemma4_profile_span(
                            "kv_read_local_decode", self._layer_config
                        ):
                            seq_lens_local, block_tables_local = (
                                _get_or_build_local_decode_aligned_metadata(
                                    attn_metadata=attn_metadata,
                                    block_tables=block_tables,
                                    seq_lens=seq_lens,
                                    local_window=ctx_window,
                                    block_size=int(k_cache.shape[1]),
                                    inf_config=inf_config,
                                )
                            )
                        # Upper bound for paged_attention_v1: prefer a cheap
                        # CPU-side scalar so the 60-layer decode loop never
                        # triggers a D->H sync here. The bound only needs to
                        # upper-cover seq_lens_local per batch row.
                        _slc_cpu = _meta_cpu_seq_lens(attn_metadata)
                        _max_cpu = _meta_cpu_max_seq_len(attn_metadata)
                        _block_sz = int(k_cache.shape[1])
                        if _slc_cpu is not None and len(_slc_cpu) > 0:
                            if ctx_window > 0:
                                max_ctx_local = max(
                                    (min(int(s), ctx_window) + _block_sz - 1)
                                    for s in _slc_cpu
                                )
                            else:
                                max_ctx_local = max(int(s) for s in _slc_cpu)
                        elif _max_cpu is not None:
                            if ctx_window > 0:
                                max_ctx_local = (
                                    min(int(_max_cpu), ctx_window) + _block_sz - 1
                                )
                            else:
                                max_ctx_local = int(_max_cpu)
                        elif _gemma4_model_policy_truthy(
                            inf_config,
                            "legacy_item_path",
                            default=False,
                        ):
                            max_ctx_local = (
                                int(torch.max(seq_lens_local).item())
                                if int(seq_lens_local.numel()) > 0
                                else 0
                            )
                        else:
                            # Conservative upper bound derived from tensor
                            # shapes (free) rather than a device-side reduce.
                            max_ctx_local = (
                                int(block_tables_local.shape[1]) * _block_sz
                                if int(seq_lens_local.numel()) > 0
                                else 0
                            )
                        with _gemma4_profile_span(
                            "attn_local_decode_kernel", self._layer_config
                        ):
                            paged_attention_v1(
                                attn_out,
                                q.reshape(
                                    bsz * seqlen, self.num_heads, self.head_dim
                                ).contiguous(),
                                k_cache,
                                v_cache,
                                self.num_heads,
                                self.scale,
                                block_tables_local,
                                seq_lens_local.to(
                                    device=q.device, dtype=seq_lens.dtype
                                ),
                                k_cache.shape[1],
                                max_ctx_local,
                                None,
                                kv_dtype_name,
                                k_scale,
                                v_scale,
                                k_scale_ptrs=k_scale_cache,
                                v_scale_ptrs=v_scale_cache,
                                num_kv_heads=self.num_kv_heads,
                                attn_scope="local",
                                layer_type=self.layer_type,
                                config=inf_config,
                                softcap=(
                                    float(softcap)
                                    if softcap is not None and float(softcap) > 0.0
                                    else None
                                ),
                                kv_select_ratio=0.0,
                                kv_select_min_blocks=0,
                            )
                        out = attn_out.view(bsz, seqlen, -1)
                    else:
                        with _gemma4_profile_span(
                            "kv_read_local_decode", self._layer_config
                        ):
                            k_ctx, v_ctx, k_positions, k_valid = (
                                _gather_recent_kv_batched(
                                    kv_cache=kv_cache,
                                    block_tables=block_tables,
                                    seq_lens=seq_lens,
                                    num_kv_heads=self.num_kv_heads,
                                    head_dim=self.head_dim,
                                    local_window=ctx_window,
                                    kv_cache_dtype=str(kv_dtype_name),
                                    inf_config=inf_config,
                                    kv_scale_cache=(k_scale_cache, v_scale_cache),
                                    seq_lens_cpu=_meta_cpu_seq_lens(attn_metadata),
                                    max_seq_len_cpu=_meta_cpu_max_seq_len(
                                        attn_metadata
                                    ),
                                )
                            )
                        q_positions = (
                            seq_lens.to(device=q.device, dtype=torch.long)[:, None]
                            - seqlen
                        ) + torch.arange(seqlen, device=q.device, dtype=torch.long)[
                            None, :
                        ]
                        out = (
                            _causal_attention_ref(
                                q.transpose(1, 2).float(),
                                k_ctx.transpose(1, 2).float(),
                                v_ctx.transpose(1, 2).float(),
                                self.scale,
                                local_window=None,
                                softcap=softcap,
                                key_padding_mask=k_valid,
                                q_positions=q_positions,
                                k_positions=k_positions,
                            )
                            .to(q.dtype)
                            .view(bsz, seqlen, -1)
                        )
            else:
                with _gemma4_profile_span("attn_global", self._layer_config):
                    from vllm.engine.lite_engine import (
                        expand_metadata_for_paged_attention,
                    )
                    from vllm.kernels.triton.paged_attention import paged_attention_v1

                    attn_out = torch.empty(
                        (bsz * seqlen, self.num_heads, self.head_dim),
                        device=q.device,
                        dtype=q.dtype,
                    )
                    block_tables = _meta_get(attn_metadata, "block_tables", None)
                    seq_lens = _meta_get(attn_metadata, "seq_lens", None)
                    use_full_ref = _should_use_full_decode_reference(
                        inf_config, str(kv_cache_dtype)
                    )
                    if (
                        use_full_ref
                        and block_tables is not None
                        and seq_lens is not None
                    ):
                        outs = []
                        _global_seq_lens_cpu = _meta_cpu_seq_lens(attn_metadata)
                        for bi in range(bsz):
                            with _gemma4_profile_span(
                                "kv_read_global_ref", self._layer_config
                            ):
                                _slc_hint = None
                                if _global_seq_lens_cpu is not None and bi < len(
                                    _global_seq_lens_cpu
                                ):
                                    _slc_hint = int(_global_seq_lens_cpu[bi])
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
                                    seq_len_cpu=_slc_hint,
                                )
                            q_i = q[bi : bi + 1].transpose(1, 2).float()
                            out_i = (
                                _causal_attention_ref(
                                    q_i,
                                    k_ctx.transpose(1, 2).float(),
                                    v_ctx.transpose(1, 2).float(),
                                    self.scale,
                                    local_window=None,
                                    softcap=softcap,
                                )
                                .to(q.dtype)
                                .view(1, seqlen, -1)
                            )
                            outs.append(out_i)
                        out = torch.cat(outs, dim=0)
                        return self.o_proj(out, lora_mapping, inf_config=inf_config)
                    seq_lens_ext, block_tables_ext = (
                        expand_metadata_for_paged_attention(
                            bsz,
                            seqlen,
                            is_prefill,
                            seq_lens,
                            block_tables,
                            q.device,
                            seq_lens_cpu=_meta_cpu_seq_lens(attn_metadata),
                        )
                    )
                    max_ctx = int(
                        max(
                            self.num_heads * self.head_dim,
                            getattr(self.config, "max_position_embeddings", 4096),
                        )
                    )
                    with _gemma4_profile_span("attn_global_kernel", self._layer_config):
                        _kv_sel_ratio = float(
                            getattr(inf_config, "kv_select_ratio", 0.0)
                            if inf_config is not None
                            else 0.0
                        )
                        _kv_sel_min = int(
                            getattr(inf_config, "kv_select_min_blocks", 4)
                            if inf_config is not None
                            else 4
                        )
                        paged_attention_v1(
                            attn_out,
                            q.reshape(
                                bsz * seqlen, self.num_heads, self.head_dim
                            ).contiguous(),
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
                            attn_scope="global",
                            layer_type=self.layer_type,
                            config=inf_config,
                            softcap=(
                                float(softcap)
                                if softcap is not None and float(softcap) > 0.0
                                else None
                            ),
                            kv_select_ratio=_kv_sel_ratio,
                            kv_select_min_blocks=_kv_sel_min,
                        )
                    out = attn_out.view(bsz, seqlen, -1)
        else:
            if self.is_kv_shared_layer:
                raise RuntimeError(
                    "KV-shared Gemma4 attention layers require a KV cache to "
                    "reuse the donor layer's key/value states."
                )
            with _gemma4_profile_span("attn_nocache", self._layer_config):
                out = _causal_attention_ref(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    self.scale,
                    local_window=local_window,
                    softcap=softcap,
                ).view(bsz, seqlen, -1)
        with _gemma4_profile_span("attn_o_proj", self._layer_config):
            return self.o_proj(out, lora_mapping, inf_config=inf_config)
