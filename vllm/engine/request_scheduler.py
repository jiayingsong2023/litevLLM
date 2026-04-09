# SPDX-License-Identifier: Apache-2.0
import asyncio
from collections import deque
from collections.abc import AsyncIterator
from typing import Any

from vllm.outputs import RequestOutput


EngineRequest = dict[str, Any]


class RequestScheduler:
    """Owns request state, stream queues, and slot lifecycle for LiteEngine."""

    def __init__(
        self,
        max_active_requests: int,
        max_queued_requests: int | None = None,
    ) -> None:
        self.max_active_requests = max_active_requests
        self.max_queued_requests = max_queued_requests or max(1, max_active_requests * 4)
        self._requests: dict[str, EngineRequest] = {}
        self._running_ids: list[str] = []
        self._queued_ids: deque[str] = deque()
        self._free_slots: list[int] = list(range(max_active_requests))
        self._request_slots: dict[str, int] = {}
        self._request_streams: dict[str, asyncio.Queue[RequestOutput | BaseException]] = {}

    @property
    def active_request_count(self) -> int:
        return len(self._running_ids) + len(self._queued_ids)

    @property
    def running_request_count(self) -> int:
        return len(self._running_ids)

    @property
    def queued_request_count(self) -> int:
        return len(self._queued_ids)

    @property
    def running_ids(self) -> list[str]:
        return list(self._running_ids)

    @property
    def queued_ids(self) -> list[str]:
        return list(self._queued_ids)

    @property
    def available_slots(self) -> int:
        return len(self._free_slots)

    def has_capacity(self) -> bool:
        return bool(self._free_slots)

    def has_queue_capacity(self) -> bool:
        return len(self._queued_ids) < self.max_queued_requests

    def allocate_slot(self) -> int:
        return self._free_slots.pop(0)

    def add_request(self, request_id: str, request: EngineRequest) -> None:
        slot_idx = request.get("slot_idx")
        self._requests[request_id] = request
        self._request_streams[request_id] = asyncio.Queue()
        if slot_idx is None:
            self._queued_ids.append(request_id)
            return
        slot_idx = int(slot_idx)
        request["slot_idx"] = slot_idx
        if slot_idx in self._free_slots:
            self._free_slots.remove(slot_idx)
        self._requests[request_id] = request
        self._request_slots[request_id] = slot_idx
        self._running_ids.append(request_id)

    def enqueue_request(self, request_id: str, request: EngineRequest) -> None:
        request = dict(request)
        request["slot_idx"] = None
        self.add_request(request_id, request)

    def admit_queued_requests(self, max_new: int | None = None) -> list[str]:
        admitted: list[str] = []
        admit_limit = len(self._free_slots) if max_new is None else max(0, int(max_new))
        while self._queued_ids and self._free_slots and len(admitted) < admit_limit:
            request_id = self._queued_ids.popleft()
            request = self._requests[request_id]
            slot_idx = self.allocate_slot()
            request["slot_idx"] = slot_idx
            self._request_slots[request_id] = slot_idx
            self._running_ids.append(request_id)
            admitted.append(request_id)
        return admitted

    def admit_specific_requests(
        self,
        request_ids: list[str],
        *,
        admitted_at: float | None = None,
    ) -> list[str]:
        admitted: list[str] = []
        for request_id in request_ids:
            if request_id not in self._queued_ids or not self._free_slots:
                continue
            self._queued_ids.remove(request_id)
            request = self._requests[request_id]
            slot_idx = self.allocate_slot()
            request["slot_idx"] = slot_idx
            request["admitted_at"] = admitted_at
            self._request_slots[request_id] = slot_idx
            self._running_ids.append(request_id)
            admitted.append(request_id)
        return admitted

    def reject_expired_queued_requests(
        self,
        *,
        now: float,
        max_queue_wait_s: float,
    ) -> list[tuple[str, str]]:
        expired: list[tuple[str, str]] = []
        if max_queue_wait_s <= 0:
            return expired
        for request_id in list(self._queued_ids):
            request = self._requests.get(request_id)
            if request is None:
                continue
            queued_at = float(request.get("queued_at") or now)
            queue_wait = now - queued_at
            if queue_wait < max_queue_wait_s:
                continue
            self._queued_ids.remove(request_id)
            self._requests.pop(request_id, None)
            self._request_slots.pop(request_id, None)
            expired.append(
                (
                    request_id,
                    f"queue timeout after {queue_wait:.3f}s (limit={max_queue_wait_s:.3f}s)",
                )
            )
        return expired

    def get_request(self, request_id: str) -> EngineRequest:
        return self._requests[request_id]

    def stream_queue(self, request_id: str) -> asyncio.Queue[RequestOutput | BaseException]:
        return self._request_streams[request_id]

    async def get_request_stream(self, request_id: str) -> AsyncIterator[RequestOutput]:
        queue = self._request_streams[request_id]
        while True:
            output = await queue.get()
            if isinstance(output, BaseException):
                raise output
            yield output
            if output.finished:
                break

    def publish_output(self, request_id: str, output: RequestOutput) -> None:
        queue = self._request_streams.get(request_id)
        if queue is not None:
            queue.put_nowait(output)

    def publish_exception(self, request_id: str, exc: BaseException) -> None:
        queue = self._request_streams.get(request_id)
        if queue is not None:
            queue.put_nowait(exc)

    def free_request(self, request_id: str) -> None:
        request = self._requests.pop(request_id, None)
        if request is not None:
            slot_idx = request.get("slot_idx")
            if slot_idx is not None:
                self._free_slots.append(int(slot_idx))
        self._request_slots.pop(request_id, None)
        if request_id in self._running_ids:
            self._running_ids.remove(request_id)
        if request_id in self._queued_ids:
            self._queued_ids.remove(request_id)

    def abort_request(self, request_id: str) -> None:
        self.free_request(request_id)

    def classify_requests(self) -> tuple[list[str], list[str]]:
        prefills: list[str] = []
        decodes: list[str] = []
        for request_id in self._running_ids:
            request = self._requests[request_id]
            if request["is_prefill"]:
                prefills.append(request_id)
            else:
                decodes.append(request_id)
        return prefills, decodes

    def request_ids(self) -> list[str]:
        return list(self._running_ids) + list(self._queued_ids)
