# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EnvScope(str, Enum):
    PUBLIC = "public"
    DEPRECATED = "deprecated"
    TOOL_ONLY = "tool_only"
    REMOVED = "removed"


@dataclass(frozen=True)
class FastInferenceEnvSpec:
    name: str
    scope: EnvScope
    replacement: str | None = None
    description: str = ""


FINAL_PUBLIC_FASTINFERENCE_ENV: frozenset[str] = frozenset(
    {
        "FASTINFERENCE_ALLOW_LEGACY_ENV",
        "FASTINFERENCE_BENCH_PROFILE",
        "FASTINFERENCE_CONFIG",
        "FASTINFERENCE_DEBUG",
        "FASTINFERENCE_KV_TYPE",
        "FASTINFERENCE_LOG_LEVEL",
        "FASTINFERENCE_PROFILE",
    }
)


def _public(name: str, description: str = "") -> FastInferenceEnvSpec:
    return FastInferenceEnvSpec(
        name=name,
        scope=EnvScope.PUBLIC,
        description=description,
    )


def _deprecated(
    name: str,
    *,
    replacement: str,
    description: str = "",
) -> FastInferenceEnvSpec:
    return FastInferenceEnvSpec(
        name=name,
        scope=EnvScope.DEPRECATED,
        replacement=replacement,
        description=description,
    )


def _tool_only(name: str, description: str = "") -> FastInferenceEnvSpec:
    return FastInferenceEnvSpec(
        name=name,
        scope=EnvScope.TOOL_ONLY,
        description=description,
    )


def _removed(name: str, description: str = "") -> FastInferenceEnvSpec:
    return FastInferenceEnvSpec(
        name=name,
        scope=EnvScope.REMOVED,
        description=description,
    )


FASTINFERENCE_ENV_REGISTRY: dict[str, FastInferenceEnvSpec] = {
    "FASTINFERENCE_ALLOW_LEGACY_ENV": _public("FASTINFERENCE_ALLOW_LEGACY_ENV"),
    "FASTINFERENCE_AWQ_CACHE_SCOPE": _deprecated(
        "FASTINFERENCE_AWQ_CACHE_SCOPE",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_DECODE_GEMV": _deprecated(
        "FASTINFERENCE_AWQ_DECODE_GEMV",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_DECODE_GEMV_BLOCK_N": _deprecated(
        "FASTINFERENCE_AWQ_DECODE_GEMV_BLOCK_N",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_DECODE_GEMV_BLOCK_PACKS": _deprecated(
        "FASTINFERENCE_AWQ_DECODE_GEMV_BLOCK_PACKS",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_DENSE_FALLBACK_CACHE": _deprecated(
        "FASTINFERENCE_AWQ_DENSE_FALLBACK_CACHE",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_DENSE_FALLBACK_MAX_GB": _deprecated(
        "FASTINFERENCE_AWQ_DENSE_FALLBACK_MAX_GB",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_FUSED_AUTOTUNE": _deprecated(
        "FASTINFERENCE_AWQ_FUSED_AUTOTUNE",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_FUSED_GATE_UP": _deprecated(
        "FASTINFERENCE_AWQ_FUSED_GATE_UP",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_FUSED_GATE_UP_BLOCK_N": _deprecated(
        "FASTINFERENCE_AWQ_FUSED_GATE_UP_BLOCK_N",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_FUSED_GATE_UP_BLOCK_PACKS": _deprecated(
        "FASTINFERENCE_AWQ_FUSED_GATE_UP_BLOCK_PACKS",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_FUSED_GEMM": _deprecated(
        "FASTINFERENCE_AWQ_FUSED_GEMM",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_FUSED_GEMM_BLOCK_K": _deprecated(
        "FASTINFERENCE_AWQ_FUSED_GEMM_BLOCK_K",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_FUSED_GEMM_BLOCK_M": _deprecated(
        "FASTINFERENCE_AWQ_FUSED_GEMM_BLOCK_M",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_FUSED_GEMM_BLOCK_N": _deprecated(
        "FASTINFERENCE_AWQ_FUSED_GEMM_BLOCK_N",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_FUSED_GEMM_DOT_BF16": _deprecated(
        "FASTINFERENCE_AWQ_FUSED_GEMM_DOT_BF16",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_FUSED_GEMM_FORCE": _deprecated(
        "FASTINFERENCE_AWQ_FUSED_GEMM_FORCE",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_FUSED_GEMM_NUM_STAGES": _deprecated(
        "FASTINFERENCE_AWQ_FUSED_GEMM_NUM_STAGES",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_FUSED_GEMM_NUM_WARPS": _deprecated(
        "FASTINFERENCE_AWQ_FUSED_GEMM_NUM_WARPS",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_FUSED_GEMM_SPLIT_K": _deprecated(
        "FASTINFERENCE_AWQ_FUSED_GEMM_SPLIT_K",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_FUSED_PROFILE_JSON": _deprecated(
        "FASTINFERENCE_AWQ_FUSED_PROFILE_JSON",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_FUSED_SCOPE": _deprecated(
        "FASTINFERENCE_AWQ_FUSED_SCOPE",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_GROUP32_GEMV_ALL": _deprecated(
        "FASTINFERENCE_AWQ_GROUP32_GEMV_ALL",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_GROUP32_GEMV_BLOCK_GROUPS": _deprecated(
        "FASTINFERENCE_AWQ_GROUP32_GEMV_BLOCK_GROUPS",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_GROUP32_GEMV_BLOCK_N": _deprecated(
        "FASTINFERENCE_AWQ_GROUP32_GEMV_BLOCK_N",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_HIGH_FIDELITY_ALL": _deprecated(
        "FASTINFERENCE_AWQ_HIGH_FIDELITY_ALL",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_HIGH_FIDELITY_PREFIXES": _deprecated(
        "FASTINFERENCE_AWQ_HIGH_FIDELITY_PREFIXES",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_HIGH_FIDELITY_PREFIX_MATCH": _deprecated(
        "FASTINFERENCE_AWQ_HIGH_FIDELITY_PREFIX_MATCH",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_LEGACY_CACHE": _deprecated(
        "FASTINFERENCE_AWQ_LEGACY_CACHE",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_MATMUL_CACHE_BEFORE_FUSED": _deprecated(
        "FASTINFERENCE_AWQ_MATMUL_CACHE_BEFORE_FUSED",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_O_PROJ_GROUP32_GEMV": _tool_only(
        "FASTINFERENCE_AWQ_O_PROJ_GROUP32_GEMV"
    ),
    "FASTINFERENCE_AWQ_O_PROJ_GROUP32_GEMV_BLOCK_GROUPS": _deprecated(
        "FASTINFERENCE_AWQ_O_PROJ_GROUP32_GEMV_BLOCK_GROUPS",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_O_PROJ_GROUP32_GEMV_BLOCK_N": _deprecated(
        "FASTINFERENCE_AWQ_O_PROJ_GROUP32_GEMV_BLOCK_N",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_O_PROJ_SPLITK": _deprecated(
        "FASTINFERENCE_AWQ_O_PROJ_SPLITK",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_O_PROJ_SPLITK_BLOCK_GROUPS": _deprecated(
        "FASTINFERENCE_AWQ_O_PROJ_SPLITK_BLOCK_GROUPS",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_O_PROJ_SPLITK_BLOCK_N": _deprecated(
        "FASTINFERENCE_AWQ_O_PROJ_SPLITK_BLOCK_N",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_O_PROJ_SPLITK_GEMV": _tool_only(
        "FASTINFERENCE_AWQ_O_PROJ_SPLITK_GEMV"
    ),
    "FASTINFERENCE_AWQ_POLICY_MATRIX": _deprecated(
        "FASTINFERENCE_AWQ_POLICY_MATRIX",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_QKV_GROUP32_GEMV_BLOCK_GROUPS": _deprecated(
        "FASTINFERENCE_AWQ_QKV_GROUP32_GEMV_BLOCK_GROUPS",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_QKV_GROUP32_GEMV_BLOCK_N": _deprecated(
        "FASTINFERENCE_AWQ_QKV_GROUP32_GEMV_BLOCK_N",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_QO_PROJ_EXACT_GEMV": _tool_only(
        "FASTINFERENCE_AWQ_QO_PROJ_EXACT_GEMV"
    ),
    "FASTINFERENCE_AWQ_Q_PROJ_GROUP32_GEMV_BLOCK_GROUPS": _deprecated(
        "FASTINFERENCE_AWQ_Q_PROJ_GROUP32_GEMV_BLOCK_GROUPS",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_AWQ_Q_PROJ_GROUP32_GEMV_BLOCK_N": _deprecated(
        "FASTINFERENCE_AWQ_Q_PROJ_GROUP32_GEMV_BLOCK_N",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_BENCH_COMPILE_CACHE_DIR": _tool_only(
        "FASTINFERENCE_BENCH_COMPILE_CACHE_DIR"
    ),
    "FASTINFERENCE_BENCH_PROFILE": _public("FASTINFERENCE_BENCH_PROFILE"),
    "FASTINFERENCE_BENCH_WARMUP_BURST_CONCURRENCY": _tool_only(
        "FASTINFERENCE_BENCH_WARMUP_BURST_CONCURRENCY"
    ),
    "FASTINFERENCE_BENCH_WARMUP_BURST_DECODE_TOKENS": _tool_only(
        "FASTINFERENCE_BENCH_WARMUP_BURST_DECODE_TOKENS"
    ),
    "FASTINFERENCE_BENCH_WARMUP_BURST_ROUNDS": _tool_only(
        "FASTINFERENCE_BENCH_WARMUP_BURST_ROUNDS"
    ),
    "FASTINFERENCE_BENCH_WARMUP_DECODE_ROUNDS": _tool_only(
        "FASTINFERENCE_BENCH_WARMUP_DECODE_ROUNDS"
    ),
    "FASTINFERENCE_BENCH_WARMUP_DECODE_TOKENS": _tool_only(
        "FASTINFERENCE_BENCH_WARMUP_DECODE_TOKENS"
    ),
    "FASTINFERENCE_BENCH_WARMUP_PREFILL_ROUNDS": _tool_only(
        "FASTINFERENCE_BENCH_WARMUP_PREFILL_ROUNDS"
    ),
    "FASTINFERENCE_BENCH_WARMUP_PRESET": _tool_only(
        "FASTINFERENCE_BENCH_WARMUP_PRESET"
    ),
    "FASTINFERENCE_BLOCK_SIZE": _deprecated(
        "FASTINFERENCE_BLOCK_SIZE",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_COMPRESSED_TENSORS_HIGH_FIDELITY": _deprecated(
        "FASTINFERENCE_COMPRESSED_TENSORS_HIGH_FIDELITY",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_CONFIG": _public("FASTINFERENCE_CONFIG"),
    "FASTINFERENCE_DEBUG": _public("FASTINFERENCE_DEBUG"),
    "FASTINFERENCE_DEEPSEEK_ATTN_FP32": _tool_only("FASTINFERENCE_DEEPSEEK_ATTN_FP32"),
    "FASTINFERENCE_DEEPSEEK_B_TIER_GREEDY": _tool_only(
        "FASTINFERENCE_DEEPSEEK_B_TIER_GREEDY"
    ),
    "FASTINFERENCE_DEEPSEEK_FP8": _deprecated(
        "FASTINFERENCE_DEEPSEEK_FP8",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_DEEPSEEK_GGUF_REGRESSION_MIN_COS": _tool_only(
        "FASTINFERENCE_DEEPSEEK_GGUF_REGRESSION_MIN_COS"
    ),
    "FASTINFERENCE_DEEPSEEK_GGUF_REGRESSION_MIN_TOPK": _tool_only(
        "FASTINFERENCE_DEEPSEEK_GGUF_REGRESSION_MIN_TOPK"
    ),
    "FASTINFERENCE_DEEPSEEK_HF_CHAT": _tool_only("FASTINFERENCE_DEEPSEEK_HF_CHAT"),
    "FASTINFERENCE_DISABLE_LINEAR_INPUT_CAP": _tool_only(
        "FASTINFERENCE_DISABLE_LINEAR_INPUT_CAP"
    ),
    "FASTINFERENCE_DISABLE_RESIDUAL_STABILIZER": _tool_only(
        "FASTINFERENCE_DISABLE_RESIDUAL_STABILIZER"
    ),
    "FASTINFERENCE_E2E_AGGRESSIVE": _tool_only("FASTINFERENCE_E2E_AGGRESSIVE"),
    "FASTINFERENCE_FUSION_LEVEL": _deprecated(
        "FASTINFERENCE_FUSION_LEVEL",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_GEMMA31B_BUCKET_CUTOFF": _tool_only(
        "FASTINFERENCE_GEMMA31B_BUCKET_CUTOFF"
    ),
    "FASTINFERENCE_GEMMA31B_BUCKET_DECODE_CUTOFF": _tool_only(
        "FASTINFERENCE_GEMMA31B_BUCKET_DECODE_CUTOFF"
    ),
    "FASTINFERENCE_GEMMA31B_BUCKET_ENABLE": _tool_only(
        "FASTINFERENCE_GEMMA31B_BUCKET_ENABLE"
    ),
    "FASTINFERENCE_GEMMA31B_BUCKET_LONG_PROFILE": _tool_only(
        "FASTINFERENCE_GEMMA31B_BUCKET_LONG_PROFILE"
    ),
    "FASTINFERENCE_GEMMA31B_BUCKET_SHORT_DECODE_PROFILE": _tool_only(
        "FASTINFERENCE_GEMMA31B_BUCKET_SHORT_DECODE_PROFILE"
    ),
    "FASTINFERENCE_GEMMA31B_BUCKET_SHORT_PROFILE": _tool_only(
        "FASTINFERENCE_GEMMA31B_BUCKET_SHORT_PROFILE"
    ),
    "FASTINFERENCE_GEMMA31B_SCHED_PROFILE": _tool_only(
        "FASTINFERENCE_GEMMA31B_SCHED_PROFILE"
    ),
    "FASTINFERENCE_GEMMA4_26B_FP32_RESIDUAL_GUARD": _tool_only(
        "FASTINFERENCE_GEMMA4_26B_FP32_RESIDUAL_GUARD"
    ),
    "FASTINFERENCE_GEMMA4_26B_FP32_RESIDUAL_GUARD_SPAN": _tool_only(
        "FASTINFERENCE_GEMMA4_26B_FP32_RESIDUAL_GUARD_SPAN"
    ),
    "FASTINFERENCE_GEMMA4_26B_FP32_RESIDUAL_GUARD_START": _tool_only(
        "FASTINFERENCE_GEMMA4_26B_FP32_RESIDUAL_GUARD_START"
    ),
    "FASTINFERENCE_GEMMA4_ALLOW_INT4_KV": _deprecated(
        "FASTINFERENCE_GEMMA4_ALLOW_INT4_KV",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_GEMMA4_C1_PRESET": _deprecated(
        "FASTINFERENCE_GEMMA4_C1_PRESET",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_GEMMA4_DENSE_DOWN_PROJ": _deprecated(
        "FASTINFERENCE_GEMMA4_DENSE_DOWN_PROJ",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_GEMMA4_DENSE_MLP": _deprecated(
        "FASTINFERENCE_GEMMA4_DENSE_MLP",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_GEMMA4_FORCE_FULL_REF_ATTN": _tool_only(
        "FASTINFERENCE_GEMMA4_FORCE_FULL_REF_ATTN"
    ),
    "FASTINFERENCE_GEMMA4_FUSED_STAGE": _deprecated(
        "FASTINFERENCE_GEMMA4_FUSED_STAGE",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_GEMMA4_LAYER_PROFILE": _deprecated(
        "FASTINFERENCE_GEMMA4_LAYER_PROFILE",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_GEMMA4_LEGACY_FP16_REF_ATTN": _tool_only(
        "FASTINFERENCE_GEMMA4_LEGACY_FP16_REF_ATTN"
    ),
    "FASTINFERENCE_GEMMA4_LEGACY_FULLPREC_KV_WRITE": _tool_only(
        "FASTINFERENCE_GEMMA4_LEGACY_FULLPREC_KV_WRITE"
    ),
    "FASTINFERENCE_GEMMA4_LEGACY_ITEM_PATH": _tool_only(
        "FASTINFERENCE_GEMMA4_LEGACY_ITEM_PATH"
    ),
    "FASTINFERENCE_GEMMA4_LOCAL_DECODE_TRITON": _tool_only(
        "FASTINFERENCE_GEMMA4_LOCAL_DECODE_TRITON"
    ),
    "FASTINFERENCE_GEMMA4_MLP_PAIR_FUSION": _tool_only(
        "FASTINFERENCE_GEMMA4_MLP_PAIR_FUSION"
    ),
    "FASTINFERENCE_GEMMA4_MOE_COMPUTE_DTYPE": _tool_only(
        "FASTINFERENCE_GEMMA4_MOE_COMPUTE_DTYPE"
    ),
    "FASTINFERENCE_GEMMA4_MOE_EXPERT_CACHE_SIZE": _tool_only(
        "FASTINFERENCE_GEMMA4_MOE_EXPERT_CACHE_SIZE"
    ),
    "FASTINFERENCE_GEMMA4_MOE_INT4_KERNEL": _tool_only(
        "FASTINFERENCE_GEMMA4_MOE_INT4_KERNEL"
    ),
    "FASTINFERENCE_GEMMA4_MOE_INT4_KERNEL_STRATEGY": _tool_only(
        "FASTINFERENCE_GEMMA4_MOE_INT4_KERNEL_STRATEGY"
    ),
    "FASTINFERENCE_GEMMA4_MOE_KERNEL_PROFILE": _deprecated(
        "FASTINFERENCE_GEMMA4_MOE_KERNEL_PROFILE",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_GEMMA4_MOE_PREFILL_GROUPED": _tool_only(
        "FASTINFERENCE_GEMMA4_MOE_PREFILL_GROUPED"
    ),
    "FASTINFERENCE_GEMMA4_MOE_PREFILL_GROUPED_STRATEGY": _tool_only(
        "FASTINFERENCE_GEMMA4_MOE_PREFILL_GROUPED_STRATEGY"
    ),
    "FASTINFERENCE_GEMMA4_ROCTX_PROFILE": _deprecated(
        "FASTINFERENCE_GEMMA4_ROCTX_PROFILE",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_GEMMA4_ROPE_CACHE_MAX_POS": _tool_only(
        "FASTINFERENCE_GEMMA4_ROPE_CACHE_MAX_POS"
    ),
    "FASTINFERENCE_GEMMA4_ROPE_CACHE_POOL_MAX": _tool_only(
        "FASTINFERENCE_GEMMA4_ROPE_CACHE_POOL_MAX"
    ),
    "FASTINFERENCE_GGUF_DEQUANT_FP32": _tool_only("FASTINFERENCE_GGUF_DEQUANT_FP32"),
    "FASTINFERENCE_GGUF_DEQUANT_FP8": _tool_only("FASTINFERENCE_GGUF_DEQUANT_FP8"),
    "FASTINFERENCE_GGUF_FP8": _deprecated(
        "FASTINFERENCE_GGUF_FP8",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_GPU_GREEDY_BYPASS_CPU_POLICIES": _tool_only(
        "FASTINFERENCE_GPU_GREEDY_BYPASS_CPU_POLICIES"
    ),
    "FASTINFERENCE_GPU_GREEDY_IGNORE_EOS": _tool_only(
        "FASTINFERENCE_GPU_GREEDY_IGNORE_EOS"
    ),
    "FASTINFERENCE_GPU_GREEDY_MAX_TOKENS_ONLY": _tool_only(
        "FASTINFERENCE_GPU_GREEDY_MAX_TOKENS_ONLY"
    ),
    "FASTINFERENCE_GPU_GREEDY_SAMPLING": _tool_only(
        "FASTINFERENCE_GPU_GREEDY_SAMPLING"
    ),
    "FASTINFERENCE_KV_FP8": _deprecated(
        "FASTINFERENCE_KV_FP8",
        replacement="FASTINFERENCE_KV_TYPE=fp8",
        description="Deprecated compatibility alias for fp8 KV cache.",
    ),
    "FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS": _deprecated(
        "FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_KV_MAX_MODEL_LEN": _deprecated(
        "FASTINFERENCE_KV_MAX_MODEL_LEN",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_KV_SELECT_MIN_BLOCKS": _deprecated(
        "FASTINFERENCE_KV_SELECT_MIN_BLOCKS",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_KV_SELECT_MIN_CONTEXT": _deprecated(
        "FASTINFERENCE_KV_SELECT_MIN_CONTEXT",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_KV_SELECT_RATIO": _deprecated(
        "FASTINFERENCE_KV_SELECT_RATIO",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_KV_SIG_DIM": _deprecated(
        "FASTINFERENCE_KV_SIG_DIM",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_KV_TYPE": _public("FASTINFERENCE_KV_TYPE"),
    "FASTINFERENCE_K_SCALE": _deprecated(
        "FASTINFERENCE_K_SCALE",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_LITE_DECODE_PRIORITY": _tool_only(
        "FASTINFERENCE_LITE_DECODE_PRIORITY"
    ),
    "FASTINFERENCE_LITE_DEFAULT_MIN_NEW_TOKENS": _tool_only(
        "FASTINFERENCE_LITE_DEFAULT_MIN_NEW_TOKENS"
    ),
    "FASTINFERENCE_LITE_PREFILL_CATCHUP_RATIO": _tool_only(
        "FASTINFERENCE_LITE_PREFILL_CATCHUP_RATIO"
    ),
    "FASTINFERENCE_LITE_PREFILL_CHUNK": _tool_only("FASTINFERENCE_LITE_PREFILL_CHUNK"),
    "FASTINFERENCE_LITE_PREFILL_MICROBATCH": _tool_only(
        "FASTINFERENCE_LITE_PREFILL_MICROBATCH"
    ),
    "FASTINFERENCE_LITE_PREFILL_RESERVED_TOKENS": _tool_only(
        "FASTINFERENCE_LITE_PREFILL_RESERVED_TOKENS"
    ),
    "FASTINFERENCE_LITE_PREFILL_RESERVE_BACKLOG": _tool_only(
        "FASTINFERENCE_LITE_PREFILL_RESERVE_BACKLOG"
    ),
    "FASTINFERENCE_LITE_QUEUE_TIMEOUT_SECONDS": _tool_only(
        "FASTINFERENCE_LITE_QUEUE_TIMEOUT_SECONDS"
    ),
    "FASTINFERENCE_LOG_LEVEL": _public("FASTINFERENCE_LOG_LEVEL"),
    "FASTINFERENCE_MAX_DECODE_STREAK": _tool_only("FASTINFERENCE_MAX_DECODE_STREAK"),
    "FASTINFERENCE_MAX_PREFILL_CHUNK": _deprecated(
        "FASTINFERENCE_MAX_PREFILL_CHUNK",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_MEM_AUDIT_TOPN": _tool_only("FASTINFERENCE_MEM_AUDIT_TOPN"),
    "FASTINFERENCE_MIN_PREFILL_CHUNK": _deprecated(
        "FASTINFERENCE_MIN_PREFILL_CHUNK",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_MOE_FP8": _deprecated(
        "FASTINFERENCE_MOE_FP8",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_MOE_LRU_SIZE": _deprecated(
        "FASTINFERENCE_MOE_LRU_SIZE",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_MOE_OFFLOAD": _deprecated(
        "FASTINFERENCE_MOE_OFFLOAD",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_MOE_PACKED_GGUF": _deprecated(
        "FASTINFERENCE_MOE_PACKED_GGUF",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_PAGED_ATTN_NUM_STAGES": _deprecated(
        "FASTINFERENCE_PAGED_ATTN_NUM_STAGES",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_PAGED_ATTN_NUM_STAGES_GLOBAL": _deprecated(
        "FASTINFERENCE_PAGED_ATTN_NUM_STAGES_GLOBAL",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_PAGED_ATTN_NUM_STAGES_LOCAL": _deprecated(
        "FASTINFERENCE_PAGED_ATTN_NUM_STAGES_LOCAL",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_PAGED_ATTN_NUM_WARPS": _deprecated(
        "FASTINFERENCE_PAGED_ATTN_NUM_WARPS",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_PAGED_ATTN_NUM_WARPS_GLOBAL": _deprecated(
        "FASTINFERENCE_PAGED_ATTN_NUM_WARPS_GLOBAL",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_PAGED_ATTN_NUM_WARPS_LOCAL": _deprecated(
        "FASTINFERENCE_PAGED_ATTN_NUM_WARPS_LOCAL",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_PREFIX_CACHE_MAX_ENTRIES": _tool_only(
        "FASTINFERENCE_PREFIX_CACHE_MAX_ENTRIES"
    ),
    "FASTINFERENCE_PROFILE": _public("FASTINFERENCE_PROFILE"),
    "FASTINFERENCE_QWEN35_FULLATTN_STABILIZER": _tool_only(
        "FASTINFERENCE_QWEN35_FULLATTN_STABILIZER"
    ),
    "FASTINFERENCE_QWEN35_FULLATTN_USE_SDP_PREFILL": _tool_only(
        "FASTINFERENCE_QWEN35_FULLATTN_USE_SDP_PREFILL"
    ),
    "FASTINFERENCE_QWEN35_MOE_FP8": _deprecated(
        "FASTINFERENCE_QWEN35_MOE_FP8",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_QWEN35_MOE_OFFLOAD": _deprecated(
        "FASTINFERENCE_QWEN35_MOE_OFFLOAD",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_QWEN35_MOE_PACKED_GGUF": _deprecated(
        "FASTINFERENCE_QWEN35_MOE_PACKED_GGUF",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_QWEN35_PROMPT_GUARD": _deprecated(
        "FASTINFERENCE_QWEN35_PROMPT_GUARD",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_QWEN35_USE_FLA_CHUNK": _tool_only(
        "FASTINFERENCE_QWEN35_USE_FLA_CHUNK"
    ),
    "FASTINFERENCE_QWEN9_AGGRESSIVE": _tool_only("FASTINFERENCE_QWEN9_AGGRESSIVE"),
    "FASTINFERENCE_QWEN9_STABLE": _tool_only("FASTINFERENCE_QWEN9_STABLE"),
    "FASTINFERENCE_SLA_TTFT_MS": _deprecated(
        "FASTINFERENCE_SLA_TTFT_MS",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_TIER_B_DEEPSEEK_GGUF_ONLY": _tool_only(
        "FASTINFERENCE_TIER_B_DEEPSEEK_GGUF_ONLY"
    ),
    "FASTINFERENCE_VERIFY_GPU_MEM_UTIL": _tool_only(
        "FASTINFERENCE_VERIFY_GPU_MEM_UTIL"
    ),
    "FASTINFERENCE_VERIFY_MAX_BATCHED_TOKENS": _tool_only(
        "FASTINFERENCE_VERIFY_MAX_BATCHED_TOKENS"
    ),
    "FASTINFERENCE_VERIFY_MAX_MODEL_LEN": _tool_only(
        "FASTINFERENCE_VERIFY_MAX_MODEL_LEN"
    ),
    "FASTINFERENCE_V_SCALE": _deprecated(
        "FASTINFERENCE_V_SCALE",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_WARMUP_DECODE_TOKENS": _deprecated(
        "FASTINFERENCE_WARMUP_DECODE_TOKENS",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_WARMUP_FIRST_REQUEST_STEPS": _deprecated(
        "FASTINFERENCE_WARMUP_FIRST_REQUEST_STEPS",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_WARMUP_PREFILL_TOKENS": _deprecated(
        "FASTINFERENCE_WARMUP_PREFILL_TOKENS",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
    "FASTINFERENCE_WARMUP_SYNC": _deprecated(
        "FASTINFERENCE_WARMUP_SYNC",
        replacement="RuntimeProfile or structured RuntimeConfig policy",
    ),
}


missing_public = FINAL_PUBLIC_FASTINFERENCE_ENV - set(FASTINFERENCE_ENV_REGISTRY)
if missing_public:
    raise RuntimeError(
        "Final public FastInference env names are not registered: "
        + ", ".join(sorted(missing_public))
    )
