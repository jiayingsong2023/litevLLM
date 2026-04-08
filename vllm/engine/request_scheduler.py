# SPDX-License-Identifier: Apache-2.0
import asyncio
from collections.abc import AsyncIterator
from typing import Any

from vllm.outputs import RequestOutput


EngineRequest = dict[str, Any]


class RequestScheduler:
    """Owns request state, stream queues, and slot lifecycle for LiteEngine."""

    def __init__(self, max_active_requests: int) -> None:
        self.max_active_requests = max_active_requests
        self._requests: dict[str, EngineRequest] = {}
        self._running_ids: list[str] = []
        self._free_slots: list[int] = list(range(max_active_requests))
        self._request_slots: dict[str, int] = {}
        self._request_streams: dict[str, asyncio.Queue[RequestOutput]] = {}

    @property
    def active_request_count(self) -> int:
        return len(self._running_ids)

    @property
    def running_ids(self) -> list[str]:
        return list(self._running_ids)

    def has_capacity(self) -> bool:
        return bool(self._free_slots)

    def allocate_slot(self) -> int:
        return self._free_slots.pop(0)

    def add_request(self, request_id: str, request: EngineRequest) -> None:
        slot_idx = int(request["slot_idx"])
        self._requests[request_id] = request
        self._request_slots[request_id] = slot_idx
        self._running_ids.append(request_id)
        self._request_streams[request_id] = asyncio.Queue()

    def get_request(self, request_id: str) -> EngineRequest:
        return self._requests[request_id]

    def stream_queue(self, request_id: str) -> asyncio.Queue[RequestOutput]:
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
            self._free_slots.append(int(request["slot_idx"]))
        self._request_slots.pop(request_id, None)
        if request_id in self._running_ids:
            self._running_ids.remove(request_id)

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
        return list(self._running_ids)
