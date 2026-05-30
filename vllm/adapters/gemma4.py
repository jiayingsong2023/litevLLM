# SPDX-License-Identifier: Apache-2.0
from typing import Any, cast

from vllm.utils.text_utils import truthy

from .base import ModelAdapter, ModelCapabilities, RuntimeModelPolicy
from .policy_keys import (
    GEMMA4_AWQ_DECODE_GEMV,
    GEMMA4_AWQ_FUSED_GATE_UP,
    GEMMA4_AWQ_FUSED_GEMM,
    GEMMA4_AWQ_FUSED_GEMM_FORCE,
    GEMMA4_AWQ_FUSED_SCOPE,
    GEMMA4_AWQ_GROUP32_GEMV_ALL,
    GEMMA4_DENSE_DOWN_PROJ,
    GEMMA4_FORCE_FULL_REF_ATTN,
    GEMMA4_FP32_RESIDUAL_GUARD_ENABLED,
    GEMMA4_FP32_RESIDUAL_GUARD_SPAN,
    GEMMA4_FP32_RESIDUAL_GUARD_START,
    GEMMA4_LEGACY_FP16_REF_ATTN,
    GEMMA4_LEGACY_FULLPREC_KV_WRITE,
    GEMMA4_LEGACY_ITEM_PATH,
    GEMMA4_LOCAL_DECODE_TRITON,
    GEMMA4_MLP_PAIR_FUSION,
    GEMMA4_MOE_BATCH_MATERIALIZE_ENABLED,
    GEMMA4_MOE_COMPUTE_DTYPE,
    GEMMA4_MOE_EXPERT_CACHE_SIZE,
    GEMMA4_MOE_INT4_KERNEL_ENABLED,
    GEMMA4_MOE_INT4_KERNEL_STRATEGY,
    GEMMA4_MOE_PREFILL_GROUPED_ENABLED,
    GEMMA4_MOE_PREFILL_GROUPED_MIN_TOKENS,
    GEMMA4_MOE_PREFILL_GROUPED_STRATEGY,
    GEMMA4_ROPE_CACHE_MAX_POS,
    GEMMA4_ROPE_CACHE_POOL_MAX,
    Gemma4ModelPolicy,
)


def _int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _supports_moe(hf_config: Any) -> bool:
    if hf_config is None:
        return False
    num_experts = _int_or(getattr(hf_config, "num_experts", 0), 0)
    top_k = _int_or(
        getattr(
            hf_config,
            "num_experts_per_tok",
            getattr(hf_config, "top_k_experts", 0),
        ),
        0,
    )
    moe_intermediate = _int_or(getattr(hf_config, "moe_intermediate_size", 0), 0)
    return num_experts > 0 and top_k > 0 and moe_intermediate > 0


class Gemma4Adapter(ModelAdapter):
    model_type = "gemma4"

    def runtime_policy(
        self,
        model_config: Any,
        runtime_config: Any,
    ) -> RuntimeModelPolicy:
        tuning_env = dict(getattr(runtime_config, "tuning_env", None) or {})
        default_stage = "all"
        fused_stage = (
            str(tuning_env.get("FASTINFERENCE_GEMMA4_FUSED_STAGE", default_stage))
            .strip()
            .lower()
        )
        if fused_stage not in ("off", "attention_only", "all"):
            fused_stage = default_stage

        kv_dtype = str(getattr(runtime_config, "kv_cache_dtype", "")).lower()
        force_kv_dtype = None
        if kv_dtype in ("turbo_int4", "int4") and not truthy(
            tuning_env.get("FASTINFERENCE_GEMMA4_ALLOW_INT4_KV")
        ):
            force_kv_dtype = "fp8"

        is_moe = _supports_moe(getattr(model_config, "hf_config", None))
        model_policy: Gemma4ModelPolicy = cast(
            Gemma4ModelPolicy,
            {
                GEMMA4_LOCAL_DECODE_TRITON: True,
                GEMMA4_FORCE_FULL_REF_ATTN: False,
                GEMMA4_LEGACY_FP16_REF_ATTN: False,
                GEMMA4_LEGACY_FULLPREC_KV_WRITE: False,
                GEMMA4_LEGACY_ITEM_PATH: False,
                GEMMA4_MLP_PAIR_FUSION: True,
                GEMMA4_FP32_RESIDUAL_GUARD_ENABLED: bool(
                    getattr(
                        runtime_config,
                        "gemma4_26b_fp32_residual_guard_enabled",
                        False,
                    )
                ),
                GEMMA4_FP32_RESIDUAL_GUARD_START: int(
                    getattr(runtime_config, "gemma4_26b_fp32_residual_guard_start", 8)
                ),
                GEMMA4_FP32_RESIDUAL_GUARD_SPAN: int(
                    getattr(runtime_config, "gemma4_26b_fp32_residual_guard_span", 3)
                ),
                GEMMA4_MOE_EXPERT_CACHE_SIZE: int(
                    getattr(runtime_config, "gemma4_moe_expert_cache_size", 32)
                ),
                GEMMA4_MOE_COMPUTE_DTYPE: str(
                    getattr(runtime_config, "gemma4_moe_compute_dtype", "auto")
                ),
                GEMMA4_MOE_INT4_KERNEL_ENABLED: bool(
                    getattr(runtime_config, "gemma4_moe_int4_kernel_enabled", True)
                ),
                GEMMA4_MOE_INT4_KERNEL_STRATEGY: str(
                    getattr(
                        runtime_config, "gemma4_moe_int4_kernel_strategy", "two_stage"
                    )
                ),
                GEMMA4_MOE_PREFILL_GROUPED_ENABLED: bool(
                    getattr(runtime_config, "gemma4_moe_prefill_grouped_enabled", False)
                ),
                GEMMA4_MOE_PREFILL_GROUPED_MIN_TOKENS: int(
                    getattr(runtime_config, "gemma4_moe_prefill_grouped_min_tokens", 17)
                ),
                GEMMA4_MOE_PREFILL_GROUPED_STRATEGY: str(
                    getattr(
                        runtime_config, "gemma4_moe_prefill_grouped_strategy", "chunked"
                    )
                ),
                GEMMA4_MOE_BATCH_MATERIALIZE_ENABLED: bool(
                    getattr(
                        runtime_config, "gemma4_moe_batch_materialize_enabled", False
                    )
                ),
                GEMMA4_ROPE_CACHE_MAX_POS: getattr(
                    runtime_config, "gemma4_rope_cache_max_pos", None
                ),
                GEMMA4_ROPE_CACHE_POOL_MAX: int(
                    getattr(runtime_config, "gemma4_rope_cache_pool_max", 8)
                ),
            },
        )
        kernel_policy = {
            GEMMA4_AWQ_FUSED_SCOPE: fused_stage,
            GEMMA4_AWQ_FUSED_GEMM: fused_stage != "off",
            GEMMA4_AWQ_FUSED_GEMM_FORCE: False,
            GEMMA4_AWQ_DECODE_GEMV: True,
            GEMMA4_AWQ_FUSED_GATE_UP: True,
        }
        if not is_moe:
            kernel_policy.update(
                {
                    GEMMA4_AWQ_GROUP32_GEMV_ALL: True,
                    GEMMA4_DENSE_DOWN_PROJ: True,
                }
            )
        tuning_env_overrides = {
            "FASTINFERENCE_AWQ_FUSED_SCOPE": fused_stage,
            "FASTINFERENCE_AWQ_FUSED_GEMM": "0" if fused_stage == "off" else "1",
            "FASTINFERENCE_AWQ_FUSED_GEMM_FORCE": "0",
            "FASTINFERENCE_AWQ_DECODE_GEMV": "1",
            "FASTINFERENCE_AWQ_FUSED_GATE_UP": "1",
        }
        if not is_moe:
            tuning_env_overrides.update(
                {
                    "FASTINFERENCE_AWQ_GROUP32_GEMV_ALL": "1",
                    "FASTINFERENCE_GEMMA4_DENSE_DOWN_PROJ": "1",
                }
            )

        return RuntimeModelPolicy(
            force_kv_cache_dtype=force_kv_dtype,
            force_kv_cache_dtype_when=("turbo_int4", "int4"),
            force_kv_cache_dtype_reason=(
                "Gemma4 accuracy guard enabled: forcing KV dtype to fp8 "
                "(set FASTINFERENCE_GEMMA4_ALLOW_INT4_KV=1 to override)."
            ),
            tuning_env_overrides=tuning_env_overrides,
            model_policy=model_policy,  # type: ignore[arg-type]
            kernel_policy=kernel_policy,
        )

    def install_tuning_config(self, tuning_env: dict[str, str]) -> None:
        from vllm.model_executor.models.gemma4 import (
            _apply_global_tuning_config,
            set_gemma4_tuning_config,
        )

        config = set_gemma4_tuning_config(tuning_env, locked=True)
        _apply_global_tuning_config(config)

    def detect(self, model: Any, model_config: Any) -> ModelCapabilities:
        hf_config = getattr(model_config, "hf_config", None)
        inner_model = getattr(model, "model", model)
        layers = getattr(inner_model, "layers", None)

        num_attention_heads = _int_or(
            getattr(hf_config, "num_attention_heads", 0)
            if hf_config is not None
            else 0,
            0,
        )
        max_kv_heads = 0
        max_head_dim = 0
        if layers is not None:
            for layer in layers:
                attn = getattr(layer, "self_attn", None)
                if attn is None:
                    continue
                max_kv_heads = max(
                    max_kv_heads,
                    _int_or(getattr(attn, "num_kv_heads", 0), 0),
                )
                max_head_dim = max(
                    max_head_dim,
                    _int_or(getattr(attn, "head_dim", 0), 0),
                )
                if num_attention_heads <= 0:
                    num_attention_heads = _int_or(getattr(attn, "num_heads", 0), 0)

        if max_kv_heads <= 0:
            max_kv_heads = _int_or(model_config.get_num_kv_heads(None), 1)
        if max_head_dim <= 0:
            hd = (
                getattr(hf_config, "global_head_dim", None)
                if hf_config is not None
                else None
            )
            if hd is None:
                hd = (
                    getattr(hf_config, "head_dim", None)
                    if hf_config is not None
                    else None
                )
            max_head_dim = _int_or(hd, _int_or(model_config.get_head_size(), 1))
        if num_attention_heads <= 0:
            num_attention_heads = _int_or(
                getattr(hf_config, "num_attention_heads", max_kv_heads)
                if hf_config is not None
                else max_kv_heads,
                max_kv_heads,
            )

        return ModelCapabilities(
            model_type=self.model_type,
            num_layers=int(model_config.get_num_layers(None)),
            num_attention_heads=num_attention_heads,
            num_kv_heads=max_kv_heads,
            head_dim=max_head_dim,
            max_model_len=int(model_config.get_max_model_len()),
            supports_moe=_supports_moe(hf_config),
            supports_fp8_kv=True,
            supports_int4_kv=True,
            supports_paged_prefill=True,
            preferred_kv_dtype="bfloat16",
        )
