# SPDX-License-Identifier: Apache-2.0
from typing import Any

from .base import ModelAdapter, ModelCapabilities, RuntimeModelPolicy


def _int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


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
        if kv_dtype in ("turbo_int4", "int4") and not _truthy(
            tuning_env.get("FASTINFERENCE_GEMMA4_ALLOW_INT4_KV")
        ):
            force_kv_dtype = "fp8"

        is_moe = _supports_moe(getattr(model_config, "hf_config", None))
        model_policy = {
            "local_decode_triton": True,
            "force_full_ref_attn": False,
            "legacy_fp16_ref_attn": False,
            "legacy_fullprec_kv_write": False,
            "legacy_item_path": False,
            "mlp_pair_fusion": True,
            "fp32_residual_guard_enabled": bool(
                getattr(
                    runtime_config,
                    "gemma4_26b_fp32_residual_guard_enabled",
                    False,
                )
            ),
            "fp32_residual_guard_start": int(
                getattr(runtime_config, "gemma4_26b_fp32_residual_guard_start", 8)
            ),
            "fp32_residual_guard_span": int(
                getattr(runtime_config, "gemma4_26b_fp32_residual_guard_span", 3)
            ),
            "moe_expert_cache_size": int(
                getattr(runtime_config, "gemma4_moe_expert_cache_size", 32)
            ),
            "moe_compute_dtype": str(
                getattr(runtime_config, "gemma4_moe_compute_dtype", "auto")
            ),
            "moe_int4_kernel_enabled": bool(
                getattr(runtime_config, "gemma4_moe_int4_kernel_enabled", True)
            ),
            "moe_int4_kernel_strategy": str(
                getattr(runtime_config, "gemma4_moe_int4_kernel_strategy", "two_stage")
            ),
            "moe_prefill_grouped_enabled": bool(
                getattr(runtime_config, "gemma4_moe_prefill_grouped_enabled", False)
            ),
            "moe_prefill_grouped_min_tokens": int(
                getattr(runtime_config, "gemma4_moe_prefill_grouped_min_tokens", 17)
            ),
            "moe_prefill_grouped_strategy": str(
                getattr(
                    runtime_config, "gemma4_moe_prefill_grouped_strategy", "chunked"
                )
            ),
            "moe_batch_materialize_enabled": bool(
                getattr(runtime_config, "gemma4_moe_batch_materialize_enabled", False)
            ),
            "rope_cache_max_pos": getattr(
                runtime_config, "gemma4_rope_cache_max_pos", None
            ),
            "rope_cache_pool_max": int(
                getattr(runtime_config, "gemma4_rope_cache_pool_max", 8)
            ),
        }
        kernel_policy = {
            "awq_fused_scope": fused_stage,
            "awq_fused_gemm": fused_stage != "off",
            "awq_fused_gemm_force": False,
            "awq_decode_gemv": True,
            "awq_fused_gate_up": True,
        }
        if not is_moe:
            kernel_policy.update(
                {
                    "awq_group32_gemv_all": True,
                    "gemma4_dense_down_proj": True,
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
            model_policy=model_policy,
            kernel_policy=kernel_policy,
        )

    def install_tuning_config(self, tuning_env: dict[str, str]) -> None:
        from vllm.model_executor.models.gemma4 import set_gemma4_tuning_config

        set_gemma4_tuning_config(tuning_env, locked=True)

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
