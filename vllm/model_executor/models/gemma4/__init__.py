# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# Re-export all symbols from sub-modules in DAG order
# to preserve the monolithic ``gemma4`` namespace.

from .config import (
    _GEMMA4_ALLOWED_TUNING_ENV,
    _GEMMA4_MOE_MATERIALIZE_BATCH_EXPERTS,
    _GEMMA4_PROFILE_ENABLED,
    _GEMMA4_PROFILE_PRINTED,
    _GEMMA4_PROFILE_STATS,
    _GEMMA4_ROCTX_PROFILE_ENABLED,
    _GEMMA4_ROPE_CACHE_POOL,
    _GEMMA4_TUNING,
    _GEMMA4_TUNING_LOCKED,
    Gemma4LayerConfig,
    _apply_global_tuning_config,
    set_gemma4_tuning_config,
)

from .policy_utils import (
    _gemma4_fp32_residual_guard_policy,
    _gemma4_kernel_policy_truthy,
    _gemma4_model_policy_truthy,
    _gemma4_policy_value,
    _get_eps,
    _meta_cpu_max_seq_len,
    _meta_cpu_seq_lens,
    _meta_get,
    _meta_set,
    _reshape_hidden_to_2d,
    _resolve_gemma4_rope_cache_max_pos,
    _resolve_gemma4_rope_cache_pool_limit,
    _resolve_max_position_plus_one_cpu,
    _restore_hidden_from_2d,
)

from .profiling import (
    _Gemma4ProfileSpan,
    _dump_gemma4_profile,
    _gemma4_profile_record,
    _gemma4_profile_span,
    _gemma4_profile_sync,
)

from .rope import (
    _get_rope,
    _get_rope_with_runtime,
    _is_local_layer,
    _layer_type_for_idx,
    Gemma4LayerRotaryEmbedding,
)

from .kv_utils import (
    _build_local_decode_aligned_metadata,
    _causal_attention_ref,
    _decode_int4_row,
    _decode_int4_rows,
    _gather_recent_kv,
    _gather_recent_kv_batched,
    _get_or_build_local_decode_aligned_metadata,
    _is_packed_or_quantized_kv_cache,
    _local_prefill_attention_sdpa,
    _repeat_kv_for_gqa,
    _should_use_full_decode_reference,
    _use_legacy_full_precision_kv_write,
    _write_full_precision_kv_cache,
)

from .mlp import Gemma4MLP

from .moe import (
    _is_gemma4_26b_a4b_like,
    _is_gemma4_moe_enabled,
    _is_gemma4_moe_layer,
    _materialize_litelinear_dense_weight_awqaware,
    _resolve_gemma4_moe_compute_dtype,
    Gemma4MoeExpertsLite,
    Gemma4SparseMoeBlock,
    Gemma4TopKRouterLite,
)

from .attention import Gemma4Attention

from .layer import (
    _residual_add_fp32,
    Gemma4DecoderLayer,
)

from .model import (
    _assert_text_only_kwargs,
    Gemma4ForCausalLM,
    Gemma4ForConditionalGeneration,
    Gemma4TextModel,
)
