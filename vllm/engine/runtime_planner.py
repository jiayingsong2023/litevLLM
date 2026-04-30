# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import torch

from vllm.adapters.base import ModelCapabilities, RuntimeModelPolicy
from vllm.engine.loadtime_policy import get_total_gpu_memory_gb
from vllm.engine.runtime_config import RuntimeConfig


def _dtype_nbytes(dtype: torch.dtype) -> int:
    f8 = getattr(torch, "float8_e4m3fn", None)
    if f8 is not None and dtype == f8:
        return 1
    if dtype == torch.uint8:
        return 1
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    if dtype == torch.float32:
        return 4
    return 2


def _align_kv_ctx_len(ctx: int, block_size: int, floor: int = 256) -> int:
    ctx = max(floor, int(ctx))
    return max(block_size, (ctx // block_size) * block_size)


@dataclass(frozen=True)
class ExecutionPlan:
    block_size: int
    max_model_len: int
    max_active_requests: int
    num_blocks_per_seq: int
    num_total_blocks: int
    step_token_budget: int
    prefill_chunk_size: int
    decode_priority_enabled: bool
    prefill_reserved_tokens: int
    prefill_reserve_backlog: int
    prefill_catchup_ratio: float
    prefill_microbatch_size: int
    is_high_end_gpu: bool


@dataclass(frozen=True)
class KVCachePlan:
    kv_dtype: torch.dtype
    kv_head_dim: int
    needs_scale_cache: bool
    theory_bytes: int


class RuntimePlanner:
    def __init__(
        self,
        runtime_config: RuntimeConfig,
        caps: ModelCapabilities,
        model_policy: RuntimeModelPolicy | None = None,
    ) -> None:
        self.runtime_config = runtime_config
        self.caps = caps
        self.model_policy = model_policy or RuntimeModelPolicy()

    def build_execution_plan(self, execution_policy_max: int) -> ExecutionPlan:
        gpu_total_gb = get_total_gpu_memory_gb()
        is_high_end_gpu = gpu_total_gb > 24.0

        block_size = self.runtime_config.block_size
        max_model_len = min(self.caps.max_model_len, self.runtime_config.max_model_len, 4096)
        if self.runtime_config.kv_max_model_len:
            max_model_len = min(max_model_len, self.runtime_config.kv_max_model_len)
        max_model_len = _align_kv_ctx_len(max_model_len, block_size)

        max_active_requests = min(execution_policy_max, self.runtime_config.max_num_seqs)
        if is_high_end_gpu and max_active_requests < 16 and self.runtime_config.kv_max_active_requests == 4:
            max_active_requests = 16
        if self.runtime_config.kv_max_active_requests != 4:
            max_active_requests = self.runtime_config.kv_max_active_requests
        max_active_requests = max(1, max_active_requests)

        num_blocks_per_seq = max_model_len // block_size
        num_total_blocks = max_active_requests * num_blocks_per_seq
        default_budget = 8192 if is_high_end_gpu else 4096
        step_token_budget = max(1, int(self.runtime_config.max_num_batched_tokens or default_budget))

        if self.runtime_config.prefill_chunk_size > 0:
            prefill_chunk_size = self.runtime_config.prefill_chunk_size
        elif is_high_end_gpu and self.model_policy.prefill_chunk_size_high_end:
            prefill_chunk_size = self.model_policy.prefill_chunk_size_high_end
        elif (not is_high_end_gpu) and self.model_policy.prefill_chunk_size_standard:
            prefill_chunk_size = self.model_policy.prefill_chunk_size_standard
        else:
            prefill_chunk_size = 1024 if is_high_end_gpu else 512

        return ExecutionPlan(
            block_size=block_size,
            max_model_len=max_model_len,
            max_active_requests=max_active_requests,
            num_blocks_per_seq=num_blocks_per_seq,
            num_total_blocks=num_total_blocks,
            step_token_budget=step_token_budget,
            prefill_chunk_size=prefill_chunk_size,
            decode_priority_enabled=self.runtime_config.enable_decode_priority,
            prefill_reserved_tokens=self.runtime_config.prefill_reserved_tokens,
            prefill_reserve_backlog=self.runtime_config.prefill_reserve_backlog,
            prefill_catchup_ratio=self.runtime_config.prefill_catchup_ratio,
            prefill_microbatch_size=self.runtime_config.prefill_microbatch_size,
            is_high_end_gpu=is_high_end_gpu,
        )

    def build_kv_cache_plan(self, execution_plan: ExecutionPlan) -> KVCachePlan:
        kv_type = self.runtime_config.kv_cache_dtype
        if kv_type == "turbo_int4" and not self.caps.supports_int4_kv:
            kv_type = "fp8"

        if kv_type == "turbo_int4":
            kv_dtype = torch.uint8
            kv_head_dim = self.caps.head_dim // 2
            needs_scale_cache = True
        elif kv_type == "fp8":
            kv_dtype = torch.float8_e4m3fn
            kv_head_dim = self.caps.head_dim
            needs_scale_cache = False
        else:
            kv_dtype = (
                torch.bfloat16
                if self.caps.preferred_kv_dtype == "bfloat16"
                else torch.float16
            )
            kv_head_dim = self.caps.head_dim
            needs_scale_cache = False

        theory_bytes = (
            self.caps.num_layers
            * 2
            * execution_plan.num_total_blocks
            * execution_plan.block_size
            * self.caps.num_kv_heads
            * kv_head_dim
            * _dtype_nbytes(kv_dtype)
        )
        return KVCachePlan(
            kv_dtype=kv_dtype,
            kv_head_dim=kv_head_dim,
            needs_scale_cache=needs_scale_cache,
            theory_bytes=theory_bytes,
        )
