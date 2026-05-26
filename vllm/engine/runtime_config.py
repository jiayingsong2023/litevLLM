# SPDX-License-Identifier: Apache-2.0
import os
from dataclasses import dataclass, field

from vllm.engine.env_registry import collect_runtime_tuning_env, get_public_env
from vllm.engine.runtime_policy import BackendRuntimePolicy, SchedulerRuntimePolicy
from vllm.engine.runtime_profile import RuntimeProfile, RuntimeProfileRegistry


@dataclass(frozen=True)
class RuntimeConfig:
    model_path: str
    tokenizer_path: str
    dtype: str
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    block_size: int
    kv_cache_dtype: str
    kv_max_model_len: int | None
    kv_max_active_requests: int
    fusion_level: int
    policy_mode: str
    enable_decode_priority: bool
    prefill_chunk_size: int
    prefill_reserved_tokens: int
    prefill_reserve_backlog: int
    prefill_catchup_ratio: float
    prefill_microbatch_size: int
    min_prefill_chunk_size: int
    max_prefill_chunk_size: int | None
    prefill_sla_ttft_ms: float
    default_min_new_tokens: int = 0
    queue_timeout_s: float = 30.0
    memory_audit_topn: int = 20
    gemma4_26b_fp32_residual_guard_enabled: bool = False
    gemma4_26b_fp32_residual_guard_start: int = 8
    gemma4_26b_fp32_residual_guard_span: int = 3
    gemma4_moe_expert_cache_size: int = 32
    gemma4_moe_compute_dtype: str = "auto"
    gemma4_moe_int4_kernel_enabled: bool = True
    gemma4_moe_int4_kernel_strategy: str = "two_stage"
    gemma4_moe_prefill_grouped_enabled: bool = False
    gemma4_moe_prefill_grouped_min_tokens: int = 17
    gemma4_moe_prefill_grouped_strategy: str = "chunked"
    gemma4_moe_batch_materialize_enabled: bool = False
    gemma4_rope_cache_max_pos: int | None = None
    gemma4_rope_cache_pool_max: int = 8
    k_scale: float = 1.0
    v_scale: float = 1.0
    use_prompt_guard: bool = True
    paged_attn_num_warps: int | None = None
    paged_attn_num_stages: int | None = None
    paged_attn_num_warps_global: int | None = None
    paged_attn_num_stages_global: int | None = None
    paged_attn_num_warps_local: int | None = None
    paged_attn_num_stages_local: int | None = None
    gemma4_c1_preset: bool = False
    tuning_env: dict[str, str] | None = None
    kv_select_ratio: float = 0.0
    kv_select_sig_dim: int = 32
    kv_select_min_blocks: int = 4
    kv_select_min_context: int = 256
    scheduler_policy: SchedulerRuntimePolicy = field(
        default_factory=SchedulerRuntimePolicy
    )
    backend_policy: BackendRuntimePolicy = field(default_factory=BackendRuntimePolicy)
    profile: RuntimeProfile | None = None

    @classmethod
    def from_vllm_config(cls, vllm_config: object) -> "RuntimeConfig":
        from vllm.engine.loadtime_policy import get_total_gpu_memory_gb

        model_config = vllm_config.model_config
        scheduler_config = vllm_config.scheduler_config

        profile = RuntimeProfileRegistry.resolve_from_env(
            model_capabilities=getattr(vllm_config, "model_capabilities", None),
            gpu_total_gb=get_total_gpu_memory_gb(),
        )
        tuning_env = collect_runtime_tuning_env(os.environ)
        tuning_env["FASTINFERENCE_PROFILE"] = profile.requested_name
        kv_cache_dtype = (
            get_public_env(os.environ, "FASTINFERENCE_KV_TYPE", profile.kv_cache_dtype)
            .strip()
            .lower()
        )
        if kv_cache_dtype == "auto":
            kv_cache_dtype = profile.kv_cache_dtype

        return cls(
            model_path=str(model_config.model),
            tokenizer_path=str(model_config.tokenizer),
            dtype=str(model_config.dtype),
            max_model_len=int(getattr(model_config, "max_model_len", 2048)),
            max_num_seqs=int(getattr(scheduler_config, "max_num_seqs", 4)),
            max_num_batched_tokens=int(
                getattr(scheduler_config, "max_num_batched_tokens", 4096)
            ),
            block_size=profile.block_size,
            kv_cache_dtype=kv_cache_dtype,
            kv_max_model_len=profile.kv_max_model_len,
            kv_max_active_requests=profile.kv_max_active_requests,
            fusion_level=profile.fusion_level,
            policy_mode=str(
                getattr(vllm_config, "runtime_policy_mode", "auto")
            ).lower(),
            enable_decode_priority=profile.enable_decode_priority,
            prefill_chunk_size=profile.prefill_chunk_size,
            prefill_reserved_tokens=profile.prefill_reserved_tokens,
            prefill_reserve_backlog=profile.prefill_reserve_backlog,
            prefill_catchup_ratio=profile.prefill_catchup_ratio,
            prefill_microbatch_size=profile.prefill_microbatch_size,
            min_prefill_chunk_size=profile.min_prefill_chunk_size,
            max_prefill_chunk_size=profile.max_prefill_chunk_size,
            prefill_sla_ttft_ms=profile.prefill_sla_ttft_ms,
            default_min_new_tokens=profile.default_min_new_tokens,
            queue_timeout_s=profile.queue_timeout_s,
            memory_audit_topn=profile.memory_audit_topn,
            k_scale=profile.k_scale,
            v_scale=profile.v_scale,
            use_prompt_guard=profile.use_prompt_guard,
            tuning_env=tuning_env,
            scheduler_policy=profile.scheduler_policy,
            backend_policy=profile.backend_policy,
            profile=profile,
        )
