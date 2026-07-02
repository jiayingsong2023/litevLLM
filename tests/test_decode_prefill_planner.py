# SPDX-License-Identifier: Apache-2.0
from vllm.engine.planners.decode_prefill_planner import DecodePrefillPlanner
from vllm.engine.request_scheduler import RequestScheduler
from vllm.engine.request_state import RequestState
from vllm.sampling_params import SamplingParams


def _make_scheduler_with_prefills(n: int):
    scheduler = RequestScheduler(max_active_requests=n)
    for i in range(n):
        req = RequestState(
            request_id=f"r{i}",
            prompt="",
            input_ids=list(range(16)),
            sampling_params=SamplingParams(),
            slot_idx=i,
            is_prefill=True,
            seq_len=0,
        )
        scheduler.add_request(f"r{i}", req)
    return scheduler


def test_build_prefill_plan_selects_first_chunk() -> None:
    planner = DecodePrefillPlanner(
        service_class_weights={"latency": 4},
        decode_service_class_quotas={},
        max_prefill_lora_adapters_per_batch=0,
        max_decode_lora_adapters_per_batch=0,
        max_prefill_multimodal_requests_per_batch=0,
        max_decode_multimodal_requests_per_batch=0,
        max_prefill_multimodal_lora_requests_per_batch=0,
        max_decode_multimodal_lora_requests_per_batch=0,
        lora_fairness_relax_threshold=0.0,
        lora_locality_tighten_threshold=0.0,
        lora_limit_relax_delta=1,
        lora_limit_tighten_delta=1,
        multimodal_prefix_cache_relax_threshold=0.2,
        multimodal_prefix_cache_tighten_threshold=0.8,
        multimodal_prefill_limit_relax_delta=1,
        multimodal_prefill_limit_tighten_delta=1,
        multimodal_lora_prefill_limit_relax_delta=1,
        multimodal_lora_prefill_limit_tighten_delta=1,
        multimodal_lora_fairness_relax_threshold=0.0,
        multimodal_lora_locality_tighten_threshold=0.0,
        prefill_chunk_size=4,
        prefill_microbatch_size=2,
    )
    scheduler = _make_scheduler_with_prefills(3)
    prefills, _ = scheduler.classify_requests()
    result = planner.build_prefill_plan(scheduler, prefills, token_budget=10)
    assert result.plan is not None
    assert result.plan.request_ids == ["r0", "r1"]
    assert result.plan.chunk_len == 4
