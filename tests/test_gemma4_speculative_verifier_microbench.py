# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Gemma4 cached verifier runner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]


def _load_module() -> Any:
    p = _ROOT / "tests" / "tools" / "gemma4_speculative_verifier_microbench.py"
    spec = importlib.util.spec_from_file_location(
        "gemma4_speculative_verifier_microbench", p
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules before exec so stringized dataclass annotations
    # can be resolved via cls.__module__.
    sys.modules["gemma4_speculative_verifier_microbench"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def verifier_mod() -> Any:
    return _load_module()


class FakeAllocator:
    """Minimal block allocator double that tracks allocated and freed IDs."""

    def __init__(self) -> None:
        self._allocated_ids: set[int] = set()
        self._freed_ids: list[int] = []

    def free(self, block_ids: list[int]) -> None:
        for bid in block_ids:
            if bid in self._allocated_ids:
                self._allocated_ids.discard(bid)
                self._freed_ids.append(bid)


class FakeKVBlockManager:
    """Minimal KVBlockManager double for CPU-side metadata tests."""

    def __init__(self, block_size: int = 16, num_blocks_per_seq: int = 8) -> None:
        self.block_size = block_size
        self.num_blocks_per_seq = num_blocks_per_seq
        self.kv_scale_caches: list[tuple[Any, Any]] = []
        self._request_blocks: dict[str, list[int]] = {}
        self._request_slot_id: dict[str, int] = {}
        self._allocator = FakeAllocator()
        self._block_table_buffer = torch.zeros((num_blocks_per_seq,), dtype=torch.int32)

    def ensure_blocks(self, request_id: str, num_tokens: int) -> int:
        needed = (int(num_tokens) + self.block_size - 1) // self.block_size
        needed = min(needed, self.num_blocks_per_seq)
        current = self._request_blocks.get(request_id, [])
        current_len = len(current)
        if needed <= current_len:
            return 0
        new_ids = list(range(current_len + 1, needed + 1))
        self._request_blocks[request_id] = current + new_ids
        self._allocator._allocated_ids.update(new_ids)
        return len(new_ids)

    def update_block_table_row(self, slot_idx: int, request_id: str) -> None:
        self._block_table_buffer.zero_()
        block_ids = self._request_blocks.get(request_id, [])
        if block_ids:
            self._block_table_buffer[: len(block_ids)] = torch.tensor(
                block_ids, dtype=torch.int32
            )

    def block_table_for_slot(self, slot_idx: int) -> torch.Tensor:
        return self._block_table_buffer


def _make_fake_kv_block_manager() -> FakeKVBlockManager:
    return FakeKVBlockManager()


class FakeRequest:
    """Minimal scheduler request double."""

    def __init__(self, request_id: str, slot_idx: int = 0, seq_len: int = 0) -> None:
        self.request_id = request_id
        self.slot_idx = slot_idx
        self.seq_len = seq_len
        self.generated_ids: list[int] = []
        self.is_prefill = False


class FakeScheduler:
    """Minimal scheduler double."""

    def __init__(self) -> None:
        self._requests: dict[str, FakeRequest] = {}

    def get_request(self, request_id: str) -> FakeRequest:
        return self._requests[request_id]

    def add_request(self, request_id: str, req: FakeRequest) -> None:
        self._requests[request_id] = req


class FakeEngine:
    """Minimal engine double for state-machine tests."""

    def __init__(self, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self.kv_block_manager = FakeKVBlockManager()
        self.scheduler = FakeScheduler()


class FakeLLM:
    """Minimal LLM double for state-machine tests."""

    def __init__(self, name: str = "fake", device: torch.device | str = "cpu") -> None:
        self.name = name
        self.engine = FakeEngine(device=device)


def _make_fake_llm_pair() -> tuple[FakeLLM, FakeLLM, FakeRequest, FakeRequest]:
    target_llm = FakeLLM("target")
    draft_llm = FakeLLM("draft")
    target_req = FakeRequest("target_req", slot_idx=0, seq_len=10)
    draft_req = FakeRequest("draft_req", slot_idx=1, seq_len=10)
    target_llm.engine.scheduler.add_request("target_req", target_req)
    draft_llm.engine.scheduler.add_request("draft_req", draft_req)
    return target_llm, draft_llm, target_req, draft_req


def _make_state(verifier_mod: Any, last_emitted: int = 100, seq_len: int = 10) -> Any:
    return verifier_mod.SpecState(
        target_seq_len=seq_len,
        draft_seq_len=seq_len,
        generated=[],
        last_emitted=last_emitted,
    )


def test_build_verifier_metadata(verifier_mod: Any) -> None:
    inf_config = SimpleNamespace(kv_type="fp8", k_scale=1.0, v_scale=1.0)
    kvbm = _make_fake_kv_block_manager()
    input_ids = torch.tensor([[10, 20, 30]])
    meta = verifier_mod.build_verifier_metadata(
        kvbm,
        inf_config,
        slot_idx=0,
        request_id="r0",
        prefix_len=5,
        input_ids=input_ids,
        num_layers=2,
    )
    assert meta["is_prefill"] is True
    assert meta["verifier_return_all_logits"] is True
    assert torch.equal(meta["seq_lens"], torch.tensor([8], dtype=torch.int32))
    assert torch.equal(meta["kv_start_indices"], torch.tensor([5], dtype=torch.int32))


def test_truncate_request_blocks_frees_tail_blocks(verifier_mod: Any) -> None:
    kvbm = _make_fake_kv_block_manager()
    kvbm._request_blocks["r0"] = [1, 2, 3, 4]
    kvbm._request_slot_id["r0"] = 0
    kvbm.block_size = 16
    kvbm._allocator._allocated_ids = {1, 2, 3, 4}

    verifier_mod._truncate_request_blocks(kvbm, "r0", new_seq_len=20)

    # 20 tokens need ceil(20/16)=2 blocks; tail blocks 3,4 should be freed.
    assert kvbm._request_blocks["r0"] == [1, 2]
    assert kvbm._allocator._allocated_ids == {1, 2}


def test_step_state_machine_first_token_reject(
    verifier_mod: Any, monkeypatch: Any
) -> None:
    target_llm, draft_llm, target_req, draft_req = _make_fake_llm_pair()

    def fake_draft(
        llm: Any, request_id: str, prefix_len: int, first_input: int, k: int
    ) -> tuple[list[int], int]:
        assert llm is draft_llm
        assert request_id == "draft_req"
        assert prefix_len == 10
        assert first_input == 100
        assert k == 3
        return [1, 2, 3], 13

    def fake_verifier(
        llm: Any, request_id: str, prefix_len: int, input_ids: torch.Tensor
    ) -> torch.Tensor:
        assert llm is target_llm
        assert request_id == "target_req"
        assert prefix_len == 10
        assert input_ids.tolist()[0] == [100, 1, 2, 3]
        logits = torch.zeros((1, 4, 1000), dtype=torch.float32)
        logits[0, 0, 999] = 100.0  # reject d_1=1, recovered=999
        return logits

    monkeypatch.setattr(verifier_mod, "run_cached_draft_k", fake_draft)
    monkeypatch.setattr(verifier_mod, "run_cached_verifier", fake_verifier)

    state = _make_state(verifier_mod)
    verifier_mod.step_state_machine(
        target_llm, draft_llm, state, 3, "target_req", "draft_req"
    )

    assert state.generated == [999]
    assert state.last_emitted == 999
    assert state.target_seq_len == 11
    assert state.draft_seq_len == 11
    assert state.target_forwards == 1
    assert state.draft_forwards == 3
    assert state.catchup_forwards == 0
    assert target_req.seq_len == 11
    assert draft_req.seq_len == 11


def test_step_state_machine_partial_accept(verifier_mod: Any, monkeypatch: Any) -> None:
    target_llm, draft_llm, target_req, draft_req = _make_fake_llm_pair()

    def fake_draft(
        llm: Any, request_id: str, prefix_len: int, first_input: int, k: int
    ) -> tuple[list[int], int]:
        return [1, 2, 3], 13

    def fake_verifier(
        llm: Any, request_id: str, prefix_len: int, input_ids: torch.Tensor
    ) -> torch.Tensor:
        logits = torch.zeros((1, 4, 1000), dtype=torch.float32)
        logits[0, 0, 1] = 100.0  # accept d_1=1
        logits[0, 1, 888] = 100.0  # reject d_2=2, recovered=888
        return logits

    monkeypatch.setattr(verifier_mod, "run_cached_draft_k", fake_draft)
    monkeypatch.setattr(verifier_mod, "run_cached_verifier", fake_verifier)

    state = _make_state(verifier_mod)
    verifier_mod.step_state_machine(
        target_llm, draft_llm, state, 3, "target_req", "draft_req"
    )

    assert state.generated == [1, 888]
    assert state.last_emitted == 888
    assert state.target_seq_len == 12
    assert state.draft_seq_len == 12
    assert state.target_forwards == 1
    assert state.draft_forwards == 3
    assert state.catchup_forwards == 0
    assert target_req.seq_len == 12
    assert draft_req.seq_len == 12


def test_step_state_machine_all_accept_catchup(
    verifier_mod: Any, monkeypatch: Any
) -> None:
    target_llm, draft_llm, target_req, draft_req = _make_fake_llm_pair()

    def fake_draft(
        llm: Any, request_id: str, prefix_len: int, first_input: int, k: int
    ) -> tuple[list[int], int]:
        if k == 3:
            assert prefix_len == 10
            assert first_input == 100
            return [1, 2, 3], 13
        # catch-up forward
        assert k == 1
        assert first_input == 3
        assert prefix_len == 13
        return [42], 14

    def fake_verifier(
        llm: Any, request_id: str, prefix_len: int, input_ids: torch.Tensor
    ) -> torch.Tensor:
        logits = torch.zeros((1, 4, 1000), dtype=torch.float32)
        logits[0, 0, 1] = 100.0
        logits[0, 1, 2] = 100.0
        logits[0, 2, 3] = 100.0
        logits[0, 3, 777] = 100.0  # bonus token
        return logits

    monkeypatch.setattr(verifier_mod, "run_cached_draft_k", fake_draft)
    monkeypatch.setattr(verifier_mod, "run_cached_verifier", fake_verifier)

    state = _make_state(verifier_mod)
    verifier_mod.step_state_machine(
        target_llm, draft_llm, state, 3, "target_req", "draft_req"
    )

    assert state.generated == [1, 2, 3, 777]
    assert state.last_emitted == 777
    assert state.target_seq_len == 14
    assert state.draft_seq_len == 14
    assert state.target_forwards == 1
    assert state.draft_forwards == 3
    assert state.catchup_forwards == 1
    assert target_req.seq_len == 14
    assert draft_req.seq_len == 14


def test_step_state_machine_k_zero_terminating_tail(
    verifier_mod: Any, monkeypatch: Any
) -> None:
    target_llm, draft_llm, target_req, draft_req = _make_fake_llm_pair()

    def fake_draft(
        llm: Any, request_id: str, prefix_len: int, first_input: int, k: int
    ) -> tuple[list[int], int]:
        raise RuntimeError("draft should not be called for k=0")

    def fake_verifier(
        llm: Any, request_id: str, prefix_len: int, input_ids: torch.Tensor
    ) -> torch.Tensor:
        assert llm is target_llm
        assert request_id == "target_req"
        assert prefix_len == 10
        assert input_ids.tolist()[0] == [100]
        logits = torch.zeros((1, 1, 1000), dtype=torch.float32)
        logits[0, 0, 555] = 100.0
        return logits

    monkeypatch.setattr(verifier_mod, "run_cached_draft_k", fake_draft)
    monkeypatch.setattr(verifier_mod, "run_cached_verifier", fake_verifier)

    state = _make_state(verifier_mod)
    verifier_mod.step_state_machine(
        target_llm, draft_llm, state, 0, "target_req", "draft_req"
    )

    assert state.generated == [555]
    assert state.last_emitted == 555
    assert state.target_seq_len == 11
    assert state.draft_seq_len == 10  # unchanged
    assert state.target_forwards == 1
    assert state.draft_forwards == 0
    assert state.catchup_forwards == 0
    assert target_req.seq_len == 11
    assert draft_req.seq_len == 10


def test_prefill_draft_persistent_keeps_request_alive(
    verifier_mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Draft prefill must keep the scheduler request alive for manual decode."""

    class _FakeRequest:
        def __init__(self, request_id: str) -> None:
            self.request_id = request_id
            self.slot_idx = 0
            self.seq_len = 5
            self.generated_ids: list[int] = [42]
            self.is_prefill = True

    class _FakeScheduler:
        def __init__(self) -> None:
            self._requests: dict[str, _FakeRequest] = {}

        def get_request(self, request_id: str) -> _FakeRequest:
            return self._requests[request_id]

        def add_request(self, request_id: str, req: _FakeRequest) -> None:
            self._requests[request_id] = req

    class _FakeOutput:
        def __init__(self, request_id: str) -> None:
            self.request_id = request_id
            self.outputs = [SimpleNamespace(token_ids=[1])]
            self.finished = False

    class _FakeEngine:
        def __init__(self, scheduler: _FakeScheduler) -> None:
            self.scheduler = scheduler

        def step(self) -> list[Any]:
            return [_FakeOutput(captured["request_id"])]

    scheduler = _FakeScheduler()
    captured: dict[str, Any] = {"request_id": None, "sampling_params": None}

    def fake_add_request(
        llm: Any,
        request_id: str,
        input_ids: list[int],
        sampling_params: Any,
    ) -> None:
        captured["request_id"] = request_id
        captured["sampling_params"] = sampling_params
        scheduler.add_request(request_id, _FakeRequest(request_id))

    monkeypatch.setattr(verifier_mod, "_add_request_with_token_ids", fake_add_request)

    fake_llm = SimpleNamespace(engine=_FakeEngine(scheduler))
    req_id = verifier_mod._prefill_draft_persistent(fake_llm, [10, 20, 30])

    assert req_id == captured["request_id"]
    assert captured["sampling_params"].ignore_eos is True

    req = scheduler.get_request(req_id)
    assert req.seq_len == 3
    assert req.generated_ids == []
    assert req.is_prefill is False
