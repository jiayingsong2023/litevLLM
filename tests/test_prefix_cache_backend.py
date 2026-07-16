# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

from vllm.engine.backend.lite_single_gpu import LiteSingleGpuBackend
from vllm.engine.request_state import RequestState
from vllm.sampling_params import SamplingParams


def test_exact_prefix_hit_transitions_request_to_decode() -> None:
    request = RequestState(
        request_id="r1",
        prompt="",
        input_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=4),
        slot_idx=0,
    )
    transition_calls: list[str] = []
    backend = object.__new__(LiteSingleGpuBackend)
    backend.kv_block_manager = SimpleNamespace(
        materialize_prefix_entry=lambda **_: None,
        update_block_table_row=lambda *_: None,
    )
    backend.scheduler = SimpleNamespace(
        transition_to_decode=transition_calls.append,
    )
    backend.prefix_cache_materialized_hits = 0
    backend.prefix_cache_materialized_saved_prefill_tokens = 0
    backend.prefix_cache_materialized_partial_hits = 0
    backend.prefix_cache_materialized_exact_hits = 0
    backend.sampling_driver = SimpleNamespace(sample_next_token=lambda *_: 7)
    completed: list[tuple[str, int]] = []
    backend._process_completion = lambda request_id, token, _: completed.append(
        (request_id, token)
    )

    LiteSingleGpuBackend._materialize_prefix_cache_entry(
        backend,
        request,
        SimpleNamespace(prompt_len=3, last_prompt_logits=None),
        prefix_len=3,
    )

    assert not request.is_prefill
    assert request.generated_ids == [7]
    assert transition_calls == ["r1"]
    assert completed == [("r1", 7)]
