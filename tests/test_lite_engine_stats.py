# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

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


def test_lite_engine_memory_audit_uses_runtime_config_topn(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_MEM_AUDIT_TOPN", "9")
    engine = LiteEngine.__new__(LiteEngine)
    engine.device = torch.device("cpu")
    engine.runtime_config = SimpleNamespace(memory_audit_topn=1)
    engine.model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 4))

    audit = engine._collect_cuda_tensor_memory_audit()

    assert audit["topn"] == 1
    assert len(audit["params_top"]) == 1
