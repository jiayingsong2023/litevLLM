# SPDX-License-Identifier: Apache-2.0
from vllm.engine.runtime_observer import InMemoryRuntimeObserver
from vllm.engine.step_plan import StepPlan


def test_inmemory_runtime_observer_records_lifecycle_events() -> None:
    observer = InMemoryRuntimeObserver()
    observer.on_request_added("r1", {"input_ids": [1, 2], "is_prefill": True})
    observer.on_request_rejected("r2", "capacity")
    observer.on_step_started(StepPlan(prefills=None, decodes=None, step_token_budget=4))
    observer.on_request_finished("r1", "eos")
    observer.on_request_aborted("r3")
    observer.on_background_error(RuntimeError("boom"), ["r4"])

    assert observer.added == ["r1"]
    assert observer.rejected == [("r2", "capacity")]
    assert observer.step_count == 1
    assert observer.finished == [("r1", "eos")]
    assert observer.aborted == ["r3"]
    assert observer.background_errors
