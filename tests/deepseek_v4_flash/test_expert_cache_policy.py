from __future__ import annotations

from vllm.model_executor.models.deepseek_v4_flash.expert_cache import (
    DeepSeekV4FlashCacheAdmissionPolicy,
    DeepSeekV4FlashCacheKey,
    DeepSeekV4FlashHotExpertPolicy,
)


def test_hot_expert_policy_pins_explicit_layer_expert_pairs() -> None:
    policy = DeepSeekV4FlashHotExpertPolicy(pinned_experts=frozenset({(1, 3), (2, 5)}))

    assert policy.is_pinned_expert(1, 3)
    assert policy.is_pinned_expert(2, 5)
    assert not policy.is_pinned_expert(1, 5)


def test_cache_admission_policy_streams_explicit_layer_expert_pairs() -> None:
    policy = DeepSeekV4FlashCacheAdmissionPolicy(
        stream_experts=frozenset({(1, 3)}),
    )

    assert not policy.should_cache_grouped_expert(layer_idx=1, expert_id=3)
    assert policy.should_cache_grouped_expert(layer_idx=1, expert_id=4)
    assert policy.should_cache_grouped_expert(layer_idx=None, expert_id=3)


def test_cache_admission_policy_rejects_all_grouped_experts_above_reuse_floor() -> None:
    policy = DeepSeekV4FlashCacheAdmissionPolicy(min_reuse_score=2)

    assert not policy.should_cache_grouped_expert(layer_idx=1, expert_id=3)


def test_cache_key_is_hashable_and_distinguishes_extra_fields() -> None:
    base = DeepSeekV4FlashCacheKey(
        namespace="grouped",
        name="blk.1.ffn_gate_exps.weight",
        device="cuda:0",
        dtype="torch.float16",
        extra=(1, 3),
    )
    same = DeepSeekV4FlashCacheKey(
        namespace="grouped",
        name="blk.1.ffn_gate_exps.weight",
        device="cuda:0",
        dtype="torch.float16",
        extra=(1, 3),
    )
    different = DeepSeekV4FlashCacheKey(
        namespace="grouped",
        name="blk.1.ffn_gate_exps.weight",
        device="cuda:0",
        dtype="torch.float16",
        extra=(1, 4),
    )

    assert base == same
    assert base != different
    assert {base: "cached"}[same] == "cached"
