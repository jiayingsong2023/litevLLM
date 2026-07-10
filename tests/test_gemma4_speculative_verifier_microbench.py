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
    spec.loader.exec_module(mod)
    sys.modules["gemma4_speculative_verifier_microbench"] = mod
    return mod


@pytest.fixture(scope="module")
def verifier_mod() -> Any:
    return _load_module()


class FakeKVBlockManager:
    """Minimal KVBlockManager double for CPU-side metadata tests."""

    def __init__(self, block_size: int = 16, num_blocks_per_seq: int = 8) -> None:
        self.block_size = block_size
        self.num_blocks_per_seq = num_blocks_per_seq
        self.kv_scale_caches: list[tuple[Any, Any]] = []
        self._request_blocks: dict[str, list[int]] = {}
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
