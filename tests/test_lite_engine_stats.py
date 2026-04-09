# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from vllm.engine.lite_engine import LiteEngine


class _FakeRuntimeController:
    def stats(self) -> dict[str, object]:
        return {
            "scheduler": {
                "active_request_count": 1,
                "running_request_count": 1,
                "queued_request_count": 0,
                "available_slots": 3,
            },
            "observer": {"step_count": 2},
            "backend": {"backend_type": "fake"},
        }

    def reset_stats(self, *, clear_prefix_cache: bool = False) -> None:
        self.reset_call = clear_prefix_cache


def test_lite_engine_stats_delegates_to_runtime_controller() -> None:
    engine = LiteEngine.__new__(LiteEngine)
    engine.runtime_controller = _FakeRuntimeController()

    assert engine.stats() == {
        "scheduler": {
            "active_request_count": 1,
            "running_request_count": 1,
            "queued_request_count": 0,
            "available_slots": 3,
        },
        "observer": {"step_count": 2},
        "backend": {"backend_type": "fake"},
    }


def test_lite_engine_reset_stats_delegates_to_runtime_controller() -> None:
    engine = LiteEngine.__new__(LiteEngine)
    engine.runtime_controller = _FakeRuntimeController()

    engine.reset_stats(clear_prefix_cache=True)

    assert engine.runtime_controller.reset_call is True
