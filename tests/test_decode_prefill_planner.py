from types import SimpleNamespace

from vllm.engine.planners.decode_prefill_planner import DecodePrefillPlanner


class _Scheduler:
    def __init__(self):
        self.requests = {
            "a": SimpleNamespace(seq_len=0, input_ids=[1, 2, 3, 4]),
            "b": SimpleNamespace(seq_len=0, input_ids=[1, 2, 3, 4]),
        }

    def get_request(self, request_id):
        return self.requests[request_id]


def test_prefill_uses_fifo_equal_length_microbatch():
    planner = DecodePrefillPlanner(prefill_chunk_size=4, prefill_microbatch_size=2)
    plan = planner.build_prefill_plan(_Scheduler(), ["a", "b"], 8).plan
    assert plan and plan.request_ids == ["a", "b"] and plan.chunk_len == 4


def test_decode_honors_verified_batch_envelope():
    planner = DecodePrefillPlanner(prefill_chunk_size=4, prefill_microbatch_size=2)
    planner.set_verified_decode_batch_sizes((1, 2))
    plan = planner.build_decode_plan(_Scheduler(), ["a", "b", "c"], 3, True).plan
    assert plan and plan.request_ids == ["a", "b"]
