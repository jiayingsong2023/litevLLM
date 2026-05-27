# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field


@dataclass
class LiteInferenceConfig:
    # Model Optimization Configuration
    # 0: No fusion
    # 1: Basic fusion (AB Gemm)
    # 2: Aggressive fusion (GateUp, KV, etc.)
    fusion_level: int

    # Engine Limits
    block_size: int
    max_model_len: int | None
    max_active_requests: int

    # Qwen-specific stability
    use_prompt_guard: bool

    # KV Cache Configuration
    kv_type: str = "auto"
    k_scale: float = 1.0
    v_scale: float = 1.0
    paged_attn_num_warps: int | None = None
    paged_attn_num_stages: int | None = None
    paged_attn_num_warps_global: int | None = None
    paged_attn_num_stages_global: int | None = None
    paged_attn_num_warps_local: int | None = None
    paged_attn_num_stages_local: int | None = None
    gemma4_c1_preset: bool = False
    tuning_env: dict[str, str] | None = None
    model_policy: dict[str, object] = field(default_factory=dict)
    kernel_policy: dict[str, object] = field(default_factory=dict)

    # KV Block Selective Attention (decode memory-bandwidth optimization)
    kv_select_ratio: float = 0.0
    kv_select_min_blocks: int = 4

    # Chunked Prefill and SLA Configuration
    min_prefill_chunk_size: int = 128
    max_prefill_chunk_size: int = 2048
    prefill_sla_ttft_ms: float = 2000.0
