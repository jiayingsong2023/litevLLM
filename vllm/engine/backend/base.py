# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Protocol

from vllm.outputs import RequestOutput


class ExecutionBackend(Protocol):
    def maybe_apply_prefix_cache(self, request_state) -> None:
        ...

    def maybe_preempt(self, step_plan, scheduler):
        ...

    def decode_step_sync(self, request_ids: list[str]) -> list[RequestOutput]:
        ...

    def run_prefills(self, step_plan, results: list[RequestOutput]) -> None:
        ...

    def run_decodes(self, step_plan, results: list[RequestOutput]) -> None:
        ...

    def stats(self) -> dict[str, object]:
        ...

    def reset_stats(self, *, clear_prefix_cache: bool = False) -> None:
        ...
