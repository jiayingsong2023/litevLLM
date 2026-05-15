# SPDX-License-Identifier: Apache-2.0
import os
from dataclasses import dataclass, field


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in ("1", "true", "yes", "on")


def _parse_kv_int_map(raw: str, *, min_value: int) -> dict[str, int] | None:
    raw = raw.strip()
    if not raw:
        return None
    out: dict[str, int] = {}
    for item in raw.split(","):
        part = item.strip()
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            continue
        try:
            out[key] = max(min_value, int(value))
        except ValueError:
            continue
    return out or None


def _parse_list(raw: str) -> set[str] | None:
    values = {item.strip() for item in raw.split(",") if item.strip()}
    return values or None


def _parse_default_min_new_tokens(raw: str) -> int:
    value = str(raw).strip().lower()
    if value in ("", "0", "false", "off", "no"):
        return 0
    try:
        return max(0, int(value))
    except ValueError:
        return 1


def _parse_memory_audit_topn(raw: str) -> int:
    try:
        return max(0, min(200, int(str(raw).strip())))
    except ValueError:
        return 20


@dataclass(frozen=True)
class SchedulerRuntimePolicy:
    max_decode_streak: int = 4
    queue_aging_threshold_s: float = 2.0
    max_prefill_deferrals: int = 2
    service_class_weights: dict[str, int] | None = None
    admission_service_class_quotas: dict[str, int] | None = None
    decode_service_class_quotas: dict[str, int] | None = None
    fairness_guardrail_queue_wait_s: float = 0.0
    fairness_guardrail_service_classes: set[str] | None = None
    max_admit_lora_adapters_per_step: int = 0
    max_prefill_lora_adapters_per_batch: int = 0
    max_decode_lora_adapters_per_batch: int = 0
    lora_fairness_relax_threshold: float = 0.0
    lora_locality_tighten_threshold: float = 0.0
    lora_limit_relax_delta: int = 1
    lora_limit_tighten_delta: int = 1
    max_admit_multimodal_per_step: int = 0
    max_prefill_multimodal_requests_per_batch: int = 0
    max_decode_multimodal_requests_per_batch: int = 0
    max_admit_multimodal_lora_per_step: int = 0
    max_prefill_multimodal_lora_requests_per_batch: int = 0
    max_decode_multimodal_lora_requests_per_batch: int = 0
    multimodal_prefix_cache_relax_threshold: float = 0.0
    multimodal_prefix_cache_tighten_threshold: float = 0.0
    multimodal_prefill_limit_relax_delta: int = 1
    multimodal_prefill_limit_tighten_delta: int = 1
    multimodal_lora_prefill_limit_relax_delta: int = 1
    multimodal_lora_prefill_limit_tighten_delta: int = 1
    multimodal_lora_fairness_relax_threshold: float = 0.0
    multimodal_lora_locality_tighten_threshold: float = 0.0

    @classmethod
    def from_env(cls) -> "SchedulerRuntimePolicy":
        return cls(
            max_decode_streak=int(
                os.environ.get("FASTINFERENCE_MAX_DECODE_STREAK", "4")
            ),
            queue_aging_threshold_s=float(
                os.environ.get("FASTINFERENCE_QUEUE_AGING_THRESHOLD_S", "2.0")
            ),
            max_prefill_deferrals=int(
                os.environ.get("FASTINFERENCE_MAX_PREFILL_DEFERRALS", "2")
            ),
            service_class_weights=_parse_kv_int_map(
                os.environ.get("FASTINFERENCE_SERVICE_CLASS_WEIGHTS", ""),
                min_value=1,
            ),
            admission_service_class_quotas=_parse_kv_int_map(
                os.environ.get("FASTINFERENCE_ADMISSION_SERVICE_CLASS_QUOTAS", ""),
                min_value=0,
            ),
            decode_service_class_quotas=_parse_kv_int_map(
                os.environ.get("FASTINFERENCE_DECODE_SERVICE_CLASS_QUOTAS", ""),
                min_value=0,
            ),
            fairness_guardrail_queue_wait_s=float(
                os.environ.get("FASTINFERENCE_FAIRNESS_GUARDRAIL_QUEUE_WAIT_S", "0.0")
            ),
            fairness_guardrail_service_classes=_parse_list(
                os.environ.get("FASTINFERENCE_FAIRNESS_GUARDRAIL_SERVICE_CLASSES", "")
            ),
            max_admit_lora_adapters_per_step=int(
                os.environ.get("FASTINFERENCE_MAX_ADMIT_LORA_ADAPTERS_PER_STEP", "0")
            ),
            max_prefill_lora_adapters_per_batch=int(
                os.environ.get("FASTINFERENCE_MAX_PREFILL_LORA_ADAPTERS_PER_BATCH", "0")
            ),
            max_decode_lora_adapters_per_batch=int(
                os.environ.get("FASTINFERENCE_MAX_DECODE_LORA_ADAPTERS_PER_BATCH", "0")
            ),
            lora_fairness_relax_threshold=float(
                os.environ.get("FASTINFERENCE_LORA_FAIRNESS_RELAX_THRESHOLD", "0.0")
            ),
            lora_locality_tighten_threshold=float(
                os.environ.get("FASTINFERENCE_LORA_LOCALITY_TIGHTEN_THRESHOLD", "0.0")
            ),
            lora_limit_relax_delta=int(
                os.environ.get("FASTINFERENCE_LORA_LIMIT_RELAX_DELTA", "1")
            ),
            lora_limit_tighten_delta=int(
                os.environ.get("FASTINFERENCE_LORA_LIMIT_TIGHTEN_DELTA", "1")
            ),
            max_admit_multimodal_per_step=int(
                os.environ.get("FASTINFERENCE_MAX_ADMIT_MULTIMODAL_PER_STEP", "0")
            ),
            max_prefill_multimodal_requests_per_batch=int(
                os.environ.get(
                    "FASTINFERENCE_MAX_PREFILL_MULTIMODAL_REQUESTS_PER_BATCH", "0"
                )
            ),
            max_decode_multimodal_requests_per_batch=int(
                os.environ.get(
                    "FASTINFERENCE_MAX_DECODE_MULTIMODAL_REQUESTS_PER_BATCH", "0"
                )
            ),
            max_admit_multimodal_lora_per_step=int(
                os.environ.get("FASTINFERENCE_MAX_ADMIT_MULTIMODAL_LORA_PER_STEP", "0")
            ),
            max_prefill_multimodal_lora_requests_per_batch=int(
                os.environ.get(
                    "FASTINFERENCE_MAX_PREFILL_MULTIMODAL_LORA_REQUESTS_PER_BATCH",
                    "0",
                )
            ),
            max_decode_multimodal_lora_requests_per_batch=int(
                os.environ.get(
                    "FASTINFERENCE_MAX_DECODE_MULTIMODAL_LORA_REQUESTS_PER_BATCH",
                    "0",
                )
            ),
            multimodal_prefix_cache_relax_threshold=float(
                os.environ.get(
                    "FASTINFERENCE_MULTIMODAL_PREFIX_CACHE_RELAX_THRESHOLD", "0.0"
                )
            ),
            multimodal_prefix_cache_tighten_threshold=float(
                os.environ.get(
                    "FASTINFERENCE_MULTIMODAL_PREFIX_CACHE_TIGHTEN_THRESHOLD", "0.0"
                )
            ),
            multimodal_prefill_limit_relax_delta=int(
                os.environ.get(
                    "FASTINFERENCE_MULTIMODAL_PREFILL_LIMIT_RELAX_DELTA", "1"
                )
            ),
            multimodal_prefill_limit_tighten_delta=int(
                os.environ.get(
                    "FASTINFERENCE_MULTIMODAL_PREFILL_LIMIT_TIGHTEN_DELTA", "1"
                )
            ),
            multimodal_lora_prefill_limit_relax_delta=int(
                os.environ.get(
                    "FASTINFERENCE_MULTIMODAL_LORA_PREFILL_LIMIT_RELAX_DELTA",
                    "1",
                )
            ),
            multimodal_lora_prefill_limit_tighten_delta=int(
                os.environ.get(
                    "FASTINFERENCE_MULTIMODAL_LORA_PREFILL_LIMIT_TIGHTEN_DELTA",
                    "1",
                )
            ),
            multimodal_lora_fairness_relax_threshold=float(
                os.environ.get(
                    "FASTINFERENCE_MULTIMODAL_LORA_FAIRNESS_RELAX_THRESHOLD",
                    "0.0",
                )
            ),
            multimodal_lora_locality_tighten_threshold=float(
                os.environ.get(
                    "FASTINFERENCE_MULTIMODAL_LORA_LOCALITY_TIGHTEN_THRESHOLD",
                    "0.0",
                )
            ),
        )


@dataclass(frozen=True)
class BackendRuntimePolicy:
    max_prefix_cache_entries: int = 8
    preemption_mode: str = "defer_prefill"
    preemption_min_backlog: int = 1
    preemption_min_decodes: int = 1
    preemption_max_queue_wait_s: float = 0.0
    preemptible_service_classes: set[str] | None = None
    preempt_multimodal_prefills: bool = False
    preempt_multimodal_max_queue_wait_s: float = 0.0
    multimodal_prefix_cache_protect_threshold: float = 0.0
    gpu_greedy_sampling: bool = False
    gpu_greedy_max_tokens_only: bool = False
    gpu_greedy_bypass_cpu_policies: bool = False
    gpu_greedy_ignore_eos: bool = False

    @classmethod
    def from_env(cls) -> "BackendRuntimePolicy":
        return cls(
            max_prefix_cache_entries=int(
                os.environ.get("FASTINFERENCE_PREFIX_CACHE_MAX_ENTRIES", "8")
            ),
            preemption_mode=os.environ.get(
                "FASTINFERENCE_PREEMPTION_MODE", "defer_prefill"
            ),
            preemption_min_backlog=int(
                os.environ.get("FASTINFERENCE_PREEMPT_MIN_BACKLOG", "1")
            ),
            preemption_min_decodes=int(
                os.environ.get("FASTINFERENCE_PREEMPT_MIN_DECODES", "1")
            ),
            preemption_max_queue_wait_s=float(
                os.environ.get("FASTINFERENCE_PREEMPT_MAX_QUEUE_WAIT_S", "0.0")
            ),
            preemptible_service_classes=_parse_list(
                os.environ.get("FASTINFERENCE_PREEMPTIBLE_SERVICE_CLASSES", "")
            ),
            preempt_multimodal_prefills=_env_truthy(
                "FASTINFERENCE_PREEMPT_MULTIMODAL_PREFILLS"
            ),
            preempt_multimodal_max_queue_wait_s=float(
                os.environ.get(
                    "FASTINFERENCE_PREEMPT_MULTIMODAL_MAX_QUEUE_WAIT_S", "0.0"
                )
            ),
            multimodal_prefix_cache_protect_threshold=float(
                os.environ.get(
                    "FASTINFERENCE_MULTIMODAL_PREFIX_CACHE_PROTECT_THRESHOLD",
                    os.environ.get(
                        "FASTINFERENCE_MULTIMODAL_PREFIX_CACHE_TIGHTEN_THRESHOLD",
                        "0.0",
                    ),
                )
            ),
            gpu_greedy_sampling=_env_truthy("FASTINFERENCE_GPU_GREEDY_SAMPLING"),
            gpu_greedy_max_tokens_only=_env_truthy(
                "FASTINFERENCE_GPU_GREEDY_MAX_TOKENS_ONLY"
            ),
            gpu_greedy_bypass_cpu_policies=_env_truthy(
                "FASTINFERENCE_GPU_GREEDY_BYPASS_CPU_POLICIES"
            ),
            gpu_greedy_ignore_eos=_env_truthy("FASTINFERENCE_GPU_GREEDY_IGNORE_EOS"),
        )


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
    default_min_new_tokens: int = 0
    queue_timeout_s: float = 30.0
    memory_audit_topn: int = 20
    gemma4_26b_fp32_residual_guard_enabled: bool = False
    gemma4_26b_fp32_residual_guard_start: int = 8
    gemma4_26b_fp32_residual_guard_span: int = 3
    gemma4_moe_expert_cache_size: int = 8
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

    @classmethod
    def from_vllm_config(cls, vllm_config: object) -> "RuntimeConfig":
        model_config = vllm_config.model_config
        scheduler_config = vllm_config.scheduler_config

        # Default to TurboQuant INT4 unless explicitly overridden.
        # Keep legacy FASTINFERENCE_KV_FP8 only when user sets KV_TYPE=auto.
        kv_type = os.environ.get("FASTINFERENCE_KV_TYPE", "turbo_int4").strip().lower()
        if kv_type == "auto":
            kv_type = (
                "fp8"
                if os.environ.get("FASTINFERENCE_KV_FP8", "0") == "1"
                else "turbo_int4"
            )

        block_size_raw = os.environ.get("FASTINFERENCE_BLOCK_SIZE", "").strip()
        block_size = int(block_size_raw) if block_size_raw else 16
        if block_size not in (16, 32, 64):
            block_size = 16

        max_model_len_env = os.environ.get("FASTINFERENCE_KV_MAX_MODEL_LEN", "").strip()
        max_active_env = os.environ.get(
            "FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS", ""
        ).strip()
        fusion_env = os.environ.get("FASTINFERENCE_FUSION_LEVEL", "").strip()
        chunk_env = os.environ.get("FASTINFERENCE_LITE_PREFILL_CHUNK", "").strip()

        def _optional_int(name: str) -> int | None:
            raw = os.environ.get(name, "").strip()
            if not raw:
                return None
            try:
                return int(raw)
            except ValueError:
                return None

        gemma4_fp32_residual_guard_start = _optional_int(
            "FASTINFERENCE_GEMMA4_26B_FP32_RESIDUAL_GUARD_START"
        )
        if gemma4_fp32_residual_guard_start is None:
            gemma4_fp32_residual_guard_start = _optional_int(
                "FASTINFERENCE_GEMMA4_26B_FP32_RESIDUAL_GUARD_LAYER"
            )
        gemma4_fp32_residual_guard_span = _optional_int(
            "FASTINFERENCE_GEMMA4_26B_FP32_RESIDUAL_GUARD_SPAN"
        )
        gemma4_moe_expert_cache_size = _optional_int(
            "FASTINFERENCE_GEMMA4_MOE_EXPERT_CACHE_SIZE"
        )
        gemma4_rope_cache_max_pos = _optional_int(
            "FASTINFERENCE_GEMMA4_ROPE_CACHE_MAX_POS"
        )
        gemma4_rope_cache_pool_max = _optional_int(
            "FASTINFERENCE_GEMMA4_ROPE_CACHE_POOL_MAX"
        )

        return cls(
            model_path=str(model_config.model),
            tokenizer_path=str(model_config.tokenizer),
            dtype=str(model_config.dtype),
            max_model_len=int(getattr(model_config, "max_model_len", 2048)),
            max_num_seqs=int(getattr(scheduler_config, "max_num_seqs", 4)),
            max_num_batched_tokens=int(
                getattr(scheduler_config, "max_num_batched_tokens", 4096)
            ),
            block_size=block_size,
            kv_cache_dtype=kv_type,
            kv_max_model_len=int(max_model_len_env) if max_model_len_env else None,
            kv_max_active_requests=int(max_active_env) if max_active_env else 4,
            fusion_level=int(fusion_env) if fusion_env else 2,
            policy_mode=str(
                getattr(vllm_config, "runtime_policy_mode", "auto")
            ).lower(),
            enable_decode_priority=(
                _env_truthy("FASTINFERENCE_LITE_DECODE_PRIORITY")
                or os.environ.get("FASTINFERENCE_LITE_DECODE_PRIORITY", "").strip()
                == ""
            ),
            prefill_chunk_size=int(chunk_env) if chunk_env else 0,
            prefill_reserved_tokens=max(
                0,
                int(os.environ.get("FASTINFERENCE_LITE_PREFILL_RESERVED_TOKENS", "0")),
            ),
            prefill_reserve_backlog=max(
                1,
                int(os.environ.get("FASTINFERENCE_LITE_PREFILL_RESERVE_BACKLOG", "2")),
            ),
            prefill_catchup_ratio=min(
                1.0,
                max(
                    0.0,
                    float(
                        os.environ.get(
                            "FASTINFERENCE_LITE_PREFILL_CATCHUP_RATIO", "0.25"
                        )
                    ),
                ),
            ),
            prefill_microbatch_size=min(
                4,
                max(
                    1, int(os.environ.get("FASTINFERENCE_LITE_PREFILL_MICROBATCH", "2"))
                ),
            ),
            default_min_new_tokens=_parse_default_min_new_tokens(
                os.environ.get("FASTINFERENCE_LITE_DEFAULT_MIN_NEW_TOKENS", "0")
            ),
            queue_timeout_s=float(
                os.environ.get("FASTINFERENCE_LITE_QUEUE_TIMEOUT_SECONDS", "30.0")
            ),
            memory_audit_topn=_parse_memory_audit_topn(
                os.environ.get("FASTINFERENCE_MEM_AUDIT_TOPN", "20")
            ),
            gemma4_26b_fp32_residual_guard_enabled=_env_truthy(
                "FASTINFERENCE_GEMMA4_26B_FP32_RESIDUAL_GUARD"
            ),
            gemma4_26b_fp32_residual_guard_start=(
                gemma4_fp32_residual_guard_start
                if gemma4_fp32_residual_guard_start is not None
                else 8
            ),
            gemma4_26b_fp32_residual_guard_span=max(
                1,
                gemma4_fp32_residual_guard_span
                if gemma4_fp32_residual_guard_span is not None
                else 3,
            ),
            gemma4_moe_expert_cache_size=max(
                0, gemma4_moe_expert_cache_size if gemma4_moe_expert_cache_size is not None else 8
            ),
            gemma4_rope_cache_max_pos=(
                max(64, gemma4_rope_cache_max_pos)
                if gemma4_rope_cache_max_pos is not None
                else None
            ),
            gemma4_rope_cache_pool_max=max(
                1,
                min(
                    128,
                    gemma4_rope_cache_pool_max
                    if gemma4_rope_cache_pool_max is not None
                    else 8,
                ),
            ),
            k_scale=float(os.environ.get("FASTINFERENCE_K_SCALE", "1.0")),
            v_scale=float(os.environ.get("FASTINFERENCE_V_SCALE", "1.0")),
            use_prompt_guard=os.environ.get("FASTINFERENCE_QWEN35_PROMPT_GUARD", "1")
            == "1",
            paged_attn_num_warps=_optional_int("FASTINFERENCE_PAGED_ATTN_NUM_WARPS"),
            paged_attn_num_stages=_optional_int("FASTINFERENCE_PAGED_ATTN_NUM_STAGES"),
            paged_attn_num_warps_global=_optional_int(
                "FASTINFERENCE_PAGED_ATTN_NUM_WARPS_GLOBAL"
            ),
            paged_attn_num_stages_global=_optional_int(
                "FASTINFERENCE_PAGED_ATTN_NUM_STAGES_GLOBAL"
            ),
            paged_attn_num_warps_local=_optional_int(
                "FASTINFERENCE_PAGED_ATTN_NUM_WARPS_LOCAL"
            ),
            paged_attn_num_stages_local=_optional_int(
                "FASTINFERENCE_PAGED_ATTN_NUM_STAGES_LOCAL"
            ),
            gemma4_c1_preset=_env_truthy("FASTINFERENCE_GEMMA4_C1_PRESET"),
            kv_select_ratio=min(
                1.0,
                max(
                    0.0,
                    float(os.environ.get("FASTINFERENCE_KV_SELECT_RATIO", "0.0")),
                ),
            ),
            kv_select_sig_dim=max(
                4,
                min(128, int(os.environ.get("FASTINFERENCE_KV_SIG_DIM", "32"))),
            ),
            kv_select_min_blocks=max(
                1,
                int(os.environ.get("FASTINFERENCE_KV_SELECT_MIN_BLOCKS", "4")),
            ),
            kv_select_min_context=max(
                64,
                int(os.environ.get("FASTINFERENCE_KV_SELECT_MIN_CONTEXT", "256")),
            ),
            tuning_env={
                key: value
                for key, value in os.environ.items()
                if key.startswith("FASTINFERENCE_")
            },
            scheduler_policy=SchedulerRuntimePolicy.from_env(),
            backend_policy=BackendRuntimePolicy.from_env(),
        )
