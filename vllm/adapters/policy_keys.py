# SPDX-License-Identifier: Apache-2.0
"""Named constants for model_policy and kernel_policy dictionary keys.

All policy keys used by model adapters MUST be defined here.
This prevents silent runtime bugs from string-key typos.
"""

# -- Gemma4 model_policy keys --
GEMMA4_LOCAL_DECODE_TRITON = "local_decode_triton"
GEMMA4_FORCE_FULL_REF_ATTN = "force_full_ref_attn"
GEMMA4_LEGACY_FP16_REF_ATTN = "legacy_fp16_ref_attn"
GEMMA4_LEGACY_FULLPREC_KV_WRITE = "legacy_fullprec_kv_write"
GEMMA4_LEGACY_ITEM_PATH = "legacy_item_path"
GEMMA4_MLP_PAIR_FUSION = "mlp_pair_fusion"
GEMMA4_FP32_RESIDUAL_GUARD_ENABLED = "fp32_residual_guard_enabled"
GEMMA4_FP32_RESIDUAL_GUARD_START = "fp32_residual_guard_start"
GEMMA4_FP32_RESIDUAL_GUARD_SPAN = "fp32_residual_guard_span"
GEMMA4_MOE_EXPERT_CACHE_SIZE = "moe_expert_cache_size"
GEMMA4_MOE_COMPUTE_DTYPE = "moe_compute_dtype"
GEMMA4_MOE_INT4_KERNEL_ENABLED = "moe_int4_kernel_enabled"
GEMMA4_MOE_INT4_KERNEL_STRATEGY = "moe_int4_kernel_strategy"
GEMMA4_MOE_PREFILL_GROUPED_ENABLED = "moe_prefill_grouped_enabled"
GEMMA4_MOE_PREFILL_GROUPED_MIN_TOKENS = "moe_prefill_grouped_min_tokens"
GEMMA4_MOE_PREFILL_GROUPED_STRATEGY = "moe_prefill_grouped_strategy"
GEMMA4_MOE_BATCH_MATERIALIZE_ENABLED = "moe_batch_materialize_enabled"
GEMMA4_ROPE_CACHE_MAX_POS = "rope_cache_max_pos"
GEMMA4_ROPE_CACHE_POOL_MAX = "rope_cache_pool_max"

# -- Gemma4 kernel_policy keys --
GEMMA4_AWQ_FUSED_SCOPE = "awq_fused_scope"
GEMMA4_AWQ_FUSED_GEMM = "awq_fused_gemm"
GEMMA4_AWQ_FUSED_GEMM_FORCE = "awq_fused_gemm_force"
GEMMA4_AWQ_DECODE_GEMV = "awq_decode_gemv"
GEMMA4_AWQ_FUSED_GATE_UP = "awq_fused_gate_up"
GEMMA4_AWQ_GROUP32_GEMV_ALL = "awq_group32_gemv_all"
GEMMA4_AWQ_ASYMMETRIC_GEMV = "awq_asymmetric_gemv"
GEMMA4_DENSE_DOWN_PROJ = "gemma4_dense_down_proj"

# -- Qwen3.5 model_policy keys --
QWEN35_FULLATTN_STABILIZER = "fullattn_stabilizer"
QWEN35_FULLATTN_USE_SDPA_PREFILL = "fullattn_use_sdpa_prefill"
QWEN35_RESIDUAL_STABILIZER = "residual_stabilizer"
QWEN35_LINEAR_INPUT_CAP = "linear_input_cap"
QWEN35_FLA_CHUNK_ENABLED = "fla_chunk_enabled"

from typing import TypedDict


class Gemma4ModelPolicy(TypedDict, total=False):
    """Typed model_policy for Gemma4 models.

    Keys match the GEMMA4_* constants defined above.
    total=False means all keys are optional.
    """

    local_decode_triton: bool
    force_full_ref_attn: bool
    legacy_fp16_ref_attn: bool
    legacy_fullprec_kv_write: bool
    legacy_item_path: bool
    mlp_pair_fusion: bool
    fp32_residual_guard_enabled: bool
    fp32_residual_guard_start: int
    fp32_residual_guard_span: int
    moe_expert_cache_size: int
    moe_compute_dtype: str
    moe_int4_kernel_enabled: bool
    moe_int4_kernel_strategy: str
    moe_prefill_grouped_enabled: bool
    moe_prefill_grouped_min_tokens: int
    moe_prefill_grouped_strategy: str
    moe_batch_materialize_enabled: bool
    rope_cache_max_pos: int | None
    rope_cache_pool_max: int


class Qwen35ModelPolicy(TypedDict, total=False):
    """Typed model_policy for Qwen3.5 models."""

    fullattn_stabilizer: bool
    fullattn_use_sdpa_prefill: bool
    residual_stabilizer: bool
    linear_input_cap: bool
    fla_chunk_enabled: bool
