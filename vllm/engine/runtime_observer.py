# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


class RuntimeObserver:
    def on_request_added(self, request_id: str, request: dict[str, Any]) -> None:
        pass

    def on_request_rejected(self, request_id: str, reason: str) -> None:
        pass

    def on_step_started(self, plan: Any) -> None:
        pass

    def on_prefill_executed(self, plan: Any, num_outputs: int) -> None:
        pass

    def on_decode_executed(self, plan: Any, num_outputs: int) -> None:
        pass

    def on_request_finished(self, request_id: str, reason: str) -> None:
        pass

    def on_request_aborted(self, request_id: str) -> None:
        pass

    def on_background_error(self, exc: BaseException, request_ids: list[str]) -> None:
        pass


class NullRuntimeObserver(RuntimeObserver):
    pass


@dataclass
class InMemoryRuntimeObserver(RuntimeObserver):
    added: list[str] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)
    finished: list[tuple[str, str]] = field(default_factory=list)
    aborted: list[str] = field(default_factory=list)
    background_errors: list[str] = field(default_factory=list)
    step_count: int = 0

    def on_request_added(self, request_id: str, request: dict[str, Any]) -> None:
        del request
        self.added.append(request_id)

    def on_request_rejected(self, request_id: str, reason: str) -> None:
        self.rejected.append((request_id, reason))

    def on_step_started(self, plan: Any) -> None:
        del plan
        self.step_count += 1

    def on_request_finished(self, request_id: str, reason: str) -> None:
        self.finished.append((request_id, reason))

    def on_request_aborted(self, request_id: str) -> None:
        self.aborted.append(request_id)

    def on_background_error(self, exc: BaseException, request_ids: list[str]) -> None:
        self.background_errors.append(f"{type(exc).__name__}:{exc}:{','.join(request_ids)}")


class LoggingRuntimeObserver(RuntimeObserver):
    def on_request_added(self, request_id: str, request: dict[str, Any]) -> None:
        logger.info(
            "runtime request added id=%s prompt_tokens=%s is_prefill=%s",
            request_id,
            len(request.get("input_ids", [])),
            request.get("is_prefill"),
        )

    def on_request_rejected(self, request_id: str, reason: str) -> None:
        logger.warning("runtime request rejected id=%s reason=%s", request_id, reason)

    def on_step_started(self, plan: Any) -> None:
        logger.debug("runtime step plan=%s", plan)

    def on_prefill_executed(self, plan: Any, num_outputs: int) -> None:
        logger.debug("runtime prefill executed plan=%s outputs=%s", plan, num_outputs)

    def on_decode_executed(self, plan: Any, num_outputs: int) -> None:
        logger.debug("runtime decode executed plan=%s outputs=%s", plan, num_outputs)

    def on_request_finished(self, request_id: str, reason: str) -> None:
        logger.info("runtime request finished id=%s reason=%s", request_id, reason)

    def on_request_aborted(self, request_id: str) -> None:
        logger.info("runtime request aborted id=%s", request_id)

    def on_background_error(self, exc: BaseException, request_ids: list[str]) -> None:
        logger.exception(
            "runtime background error requests=%s exc=%s", request_ids, exc
        )
