# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

DEEPSEEK_V4_FLASH_TARGET_GGUF_BYTES = 86_720_111_488
DEEPSEEK_V4_FLASH_DEFAULT_UMA_BUDGET_BYTES = 61 * 1024 * 1024 * 1024
DEEPSEEK_V4_FLASH_MIN_SYSTEM_HEADROOM_BYTES = 1536 * 1024 * 1024


@dataclass(frozen=True)
class DeepSeekV4FlashShape:
    num_layers: int = 43
    hidden_size: int = 4096
    vocab_size: int = 129280
    num_attention_heads: int = 64
    num_kv_heads: int = 1
    head_dim: int = 512
    value_dim: int = 512
    rotary_dim: int = 64
    output_groups: int = 8
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    num_experts: int = 256
    num_experts_per_tok: int = 6
    num_shared_experts: int = 1
    expert_intermediate_size: int = 2048
    sliding_window: int = 128
    indexer_heads: int = 64
    indexer_head_dim: int = 128
    indexer_top_k: int = 512


@dataclass(frozen=True)
class DeepSeekV4FlashContextEstimate:
    context_length: int
    raw_kv_bytes: int
    compressed_kv_bytes: int
    scratch_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.raw_kv_bytes + self.compressed_kv_bytes + self.scratch_bytes


@dataclass(frozen=True)
class DeepSeekV4FlashRuntimeBudget:
    context: DeepSeekV4FlashContextEstimate
    model_mmap_bytes: int
    resident_weight_bytes: int
    expert_cache_bytes: int
    uma_budget_bytes: int
    min_system_headroom_bytes: int

    @property
    def resident_bytes(self) -> int:
        return (
            self.context.total_bytes
            + self.resident_weight_bytes
            + self.expert_cache_bytes
        )

    @property
    def available_headroom_bytes(self) -> int:
        return self.uma_budget_bytes - self.resident_bytes

    @property
    def has_required_headroom(self) -> bool:
        return self.available_headroom_bytes >= self.min_system_headroom_bytes


DEEPSEEK_V4_FLASH_SHAPE = DeepSeekV4FlashShape()


def layer_compress_ratio(layer_idx: int) -> int:
    if layer_idx < 0 or layer_idx >= DEEPSEEK_V4_FLASH_SHAPE.num_layers:
        raise ValueError(f"layer index out of range: {layer_idx}")
    if layer_idx < 2:
        return 0
    return 4 if layer_idx % 2 == 0 else 128


class DeepSeekV4FlashMemoryPolicy:
    max_first_release_context: int = 8192
    default_expert_cache_bytes: int = 2 * 1024 * 1024 * 1024
    default_uma_budget_bytes: int = DEEPSEEK_V4_FLASH_DEFAULT_UMA_BUDGET_BYTES
    min_system_headroom_bytes: int = DEEPSEEK_V4_FLASH_MIN_SYSTEM_HEADROOM_BYTES

    def __init__(
        self,
        *,
        max_first_release_context: int | None = None,
        default_expert_cache_bytes: int | None = None,
        default_uma_budget_bytes: int | None = None,
        min_system_headroom_bytes: int | None = None,
    ) -> None:
        if max_first_release_context is not None:
            self.max_first_release_context = max_first_release_context
        if default_expert_cache_bytes is not None:
            if default_expert_cache_bytes < 0:
                raise ValueError("expert cache bytes must be non-negative")
            self.default_expert_cache_bytes = default_expert_cache_bytes
        if default_uma_budget_bytes is not None:
            if default_uma_budget_bytes <= 0:
                raise ValueError("UMA budget bytes must be positive")
            self.default_uma_budget_bytes = default_uma_budget_bytes
        if min_system_headroom_bytes is not None:
            if min_system_headroom_bytes < 0:
                raise ValueError("minimum system headroom bytes must be non-negative")
            self.min_system_headroom_bytes = min_system_headroom_bytes

    def validate_context_length(self, context_length: int) -> int:
        if context_length <= 0:
            raise ValueError("context length must be positive")
        if context_length > self.max_first_release_context:
            raise ValueError(
                "DeepSeek V4 Flash first release supports context <= "
                f"{self.max_first_release_context}; got {context_length}"
            )
        return context_length

    def estimate_context_bytes(
        self,
        context_length: int,
        *,
        prefill_cap: int = 4096,
    ) -> DeepSeekV4FlashContextEstimate:
        ctx = self.validate_context_length(context_length)
        shape = DEEPSEEK_V4_FLASH_SHAPE
        raw_cap = min(
            max(shape.sliding_window + prefill_cap, shape.sliding_window),
            ctx,
        )
        raw_cap = min(((raw_cap + 255) // 256) * 256, ctx)
        raw_kv = shape.num_layers * raw_cap * shape.head_dim * 4
        compressed_kv = 0
        for layer_idx in range(shape.num_layers):
            ratio = layer_compress_ratio(layer_idx)
            if ratio == 0:
                continue
            comp_cap = ctx // ratio + 2
            compressed_kv += comp_cap * shape.head_dim * 2
            if ratio == 4:
                compressed_kv += comp_cap * shape.indexer_head_dim * 4
        scratch = 2 * (ctx // 4 + 2) * prefill_cap * 4
        return DeepSeekV4FlashContextEstimate(
            context_length=ctx,
            raw_kv_bytes=raw_kv,
            compressed_kv_bytes=compressed_kv,
            scratch_bytes=scratch,
        )

    def estimate_runtime_budget(
        self,
        context_length: int,
        *,
        model_mmap_bytes: int = DEEPSEEK_V4_FLASH_TARGET_GGUF_BYTES,
        resident_weight_bytes: int = 0,
        expert_cache_bytes: int | None = None,
        uma_budget_bytes: int | None = None,
    ) -> DeepSeekV4FlashRuntimeBudget:
        if model_mmap_bytes <= 0:
            raise ValueError("model mmap bytes must be positive")
        if resident_weight_bytes < 0:
            raise ValueError("resident weight bytes must be non-negative")
        if expert_cache_bytes is None:
            expert_cache_bytes = self.default_expert_cache_bytes
        if expert_cache_bytes < 0:
            raise ValueError("expert cache bytes must be non-negative")
        if uma_budget_bytes is None:
            uma_budget_bytes = self.default_uma_budget_bytes
        if uma_budget_bytes <= 0:
            raise ValueError("UMA budget bytes must be positive")

        return DeepSeekV4FlashRuntimeBudget(
            context=self.estimate_context_bytes(context_length),
            model_mmap_bytes=model_mmap_bytes,
            resident_weight_bytes=resident_weight_bytes,
            expert_cache_bytes=expert_cache_bytes,
            uma_budget_bytes=uma_budget_bytes,
            min_system_headroom_bytes=self.min_system_headroom_bytes,
        )

    def validate_runtime_budget(
        self,
        budget: DeepSeekV4FlashRuntimeBudget,
    ) -> DeepSeekV4FlashRuntimeBudget:
        if not budget.has_required_headroom:
            raise ValueError(
                "insufficient DeepSeek V4 Flash UMA headroom: "
                f"resident={budget.resident_bytes} bytes, "
                f"budget={budget.uma_budget_bytes} bytes, "
                f"available={budget.available_headroom_bytes} bytes, "
                f"required={budget.min_system_headroom_bytes} bytes"
            )
        return budget
