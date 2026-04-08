# SPDX-License-Identifier: Apache-2.0
import os
from dataclasses import dataclass
from typing import Optional


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in ("1", "true", "yes", "on")


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
    kv_max_model_len: Optional[int]
    kv_max_active_requests: int
    fusion_level: int
    policy_mode: str
    enable_decode_priority: bool
    prefill_chunk_size: int
    prefill_reserved_tokens: int
    prefill_reserve_backlog: int
    prefill_catchup_ratio: float
    prefill_microbatch_size: int
    k_scale: float = 1.0
    v_scale: float = 1.0
    use_prompt_guard: bool = True

    @classmethod
    def from_vllm_config(cls, vllm_config: object) -> "RuntimeConfig":
        model_config = vllm_config.model_config
        scheduler_config = vllm_config.scheduler_config

        kv_type = os.environ.get("FASTINFERENCE_KV_TYPE", "auto").strip().lower()
        if kv_type == "auto":
            kv_type = "fp8" if os.environ.get("FASTINFERENCE_KV_FP8", "0") == "1" else "turbo_int4"

        block_size_raw = os.environ.get("FASTINFERENCE_BLOCK_SIZE", "").strip()
        block_size = int(block_size_raw) if block_size_raw else 16
        if block_size not in (16, 32, 64):
            block_size = 16

        max_model_len_env = os.environ.get("FASTINFERENCE_KV_MAX_MODEL_LEN", "").strip()
        max_active_env = os.environ.get("FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS", "").strip()
        fusion_env = os.environ.get("FASTINFERENCE_FUSION_LEVEL", "").strip()
        chunk_env = os.environ.get("FASTINFERENCE_LITE_PREFILL_CHUNK", "").strip()

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
            policy_mode=str(getattr(vllm_config, "runtime_policy_mode", "auto")).lower(),
            enable_decode_priority=(
                _env_truthy("FASTINFERENCE_LITE_DECODE_PRIORITY")
                or os.environ.get("FASTINFERENCE_LITE_DECODE_PRIORITY", "").strip() == ""
            ),
            prefill_chunk_size=int(chunk_env) if chunk_env else 0,
            prefill_reserved_tokens=max(
                0, int(os.environ.get("FASTINFERENCE_LITE_PREFILL_RESERVED_TOKENS", "0"))
            ),
            prefill_reserve_backlog=max(
                1, int(os.environ.get("FASTINFERENCE_LITE_PREFILL_RESERVE_BACKLOG", "2"))
            ),
            prefill_catchup_ratio=min(
                1.0,
                max(0.0, float(os.environ.get("FASTINFERENCE_LITE_PREFILL_CATCHUP_RATIO", "0.25"))),
            ),
            prefill_microbatch_size=min(
                4, max(1, int(os.environ.get("FASTINFERENCE_LITE_PREFILL_MICROBATCH", "2")))
            ),
            k_scale=float(os.environ.get("FASTINFERENCE_K_SCALE", "1.0")),
            v_scale=float(os.environ.get("FASTINFERENCE_V_SCALE", "1.0")),
            use_prompt_guard=os.environ.get("FASTINFERENCE_QWEN35_PROMPT_GUARD", "1") == "1",
        )
