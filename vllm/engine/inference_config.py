# SPDX-License-Identifier: Apache-2.0
import os
from dataclasses import dataclass


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

    # KV Block Selective Attention (decode memory-bandwidth optimization)
    kv_select_ratio: float = 0.0
    kv_select_sig_dim: int = 32
    kv_select_min_blocks: int = 4
    kv_select_min_context: int = 256

    # Chunked Prefill and SLA Configuration
    min_prefill_chunk_size: int = 128
    max_prefill_chunk_size: int = 2048
    prefill_sla_ttft_ms: float = 2000.0

    @classmethod
    def from_env(cls) -> "LiteInferenceConfig":
        # Resolve KV Type.
        # Default is TurboQuant INT4; legacy FP8 toggle applies only in auto mode.
        kv_type = os.environ.get("FASTINFERENCE_KV_TYPE", "turbo_int4")
        kv_fp8_env = os.environ.get("FASTINFERENCE_KV_FP8", "0")

        if kv_type == "auto":
            kv_type = "fp8" if kv_fp8_env == "1" else "turbo_int4"

        # Resolve Fusion Level (Consolidating QWEN35_FUSED_*)
        # Default to level 2 (Aggressive) if not set
        fusion_level_env = os.environ.get("FASTINFERENCE_FUSION_LEVEL")
        fusion_level = int(fusion_level_env) if fusion_level_env is not None else 2

        block_size_env = os.environ.get("FASTINFERENCE_BLOCK_SIZE", "").strip()
        block_size = int(block_size_env) if block_size_env else 16
        if block_size not in [16, 32, 64]:
            print(
                f">>>> [Warning] Unsupported block_size {block_size}. Defaulting to 16."
            )
            block_size = 16

        max_model_len_env = os.environ.get("FASTINFERENCE_KV_MAX_MODEL_LEN", "").strip()
        max_model_len = int(max_model_len_env) if max_model_len_env else None

        max_active_requests = int(
            os.environ.get("FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS", "4")
        )

        def _optional_int(name: str) -> int | None:
            raw = os.environ.get(name, "").strip()
            if not raw:
                return None
            try:
                return int(raw)
            except ValueError:
                return None

        return cls(
            kv_type=kv_type,
            k_scale=float(os.environ.get("FASTINFERENCE_K_SCALE", "1.0")),
            v_scale=float(os.environ.get("FASTINFERENCE_V_SCALE", "1.0")),
            fusion_level=fusion_level,
            block_size=block_size,
            max_model_len=max_model_len,
            max_active_requests=max_active_requests,
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
            gemma4_c1_preset=os.environ.get("FASTINFERENCE_GEMMA4_C1_PRESET", "")
            .strip()
            .lower()
            in ("1", "true", "yes", "on"),
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
            min_prefill_chunk_size=int(
                os.environ.get("FASTINFERENCE_MIN_PREFILL_CHUNK", "128")
            ),
            max_prefill_chunk_size=int(
                os.environ.get("FASTINFERENCE_MAX_PREFILL_CHUNK", "2048")
            ),
            prefill_sla_ttft_ms=float(
                os.environ.get("FASTINFERENCE_SLA_TTFT_MS", "2000.0")
            ),
            tuning_env={
                key: value
                for key, value in os.environ.items()
                if key.startswith("FASTINFERENCE_")
            },
        )
