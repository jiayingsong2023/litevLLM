# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib
import threading
from collections import deque
from collections.abc import AsyncIterator

from vllm.engine.request_state import RequestState
from vllm.outputs import RequestOutput


class RequestScheduler:
    """Owns request state, stream queues, and slot lifecycle for LiteEngine."""

    def __init__(
        self,
        max_active_requests: int,
        max_queued_requests: int | None = None,
    ) -> None:
        self.max_active_requests = max_active_requests
        self.max_queued_requests = max_queued_requests or max(
            1, max_active_requests * 4
        )
        self._requests: dict[str, RequestState] = {}
        self._running_ids: list[str] = []
        self._running_ids_set: set[str] = set()
        self._queued_ids: deque[str] = deque()
        self._queued_ids_set: set[str] = set()
        self._prefill_ids: set[str] = set()
        self._decode_ids: set[str] = set()
        self._free_slots: deque[int] = deque(range(max_active_requests))
        self._request_slots: dict[str, int] = {}
        self._request_streams: dict[
            str, asyncio.Queue[RequestOutput | BaseException]
        ] = {}
        self._event_loop: asyncio.AbstractEventLoop | None = None
        with contextlib.suppress(RuntimeError):
            self._event_loop = asyncio.get_running_loop()
        self._event_loop_thread_id: int | None = threading.current_thread().ident

    def _set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the asyncio event loop that owns this scheduler's stream queues.

        Called by AsyncDriver when it starts in an async context. This lets
        publish_output/publish_exception safely dispatch from a background
        worker thread back to the event loop thread.
        """
        self._event_loop = loop
        self._event_loop_thread_id = threading.current_thread().ident

    def _is_event_loop_thread(self) -> bool:
        return threading.current_thread().ident == self._event_loop_thread_id

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
        return self._free_slots.popleft()

    def _add_running(self, request_id: str, request: RequestState) -> None:
        """Register a request as running and update prefill/decode indexes."""
        self._running_ids.append(request_id)
        self._running_ids_set.add(request_id)
        if request.is_prefill:
            self._prefill_ids.add(request_id)
        else:
            self._decode_ids.add(request_id)

    def _remove_from_indexes(self, request_id: str) -> None:
        self._running_ids_set.discard(request_id)
        self._queued_ids_set.discard(request_id)
        self._prefill_ids.discard(request_id)
        self._decode_ids.discard(request_id)

    def add_request(
        self,
        request_id: str,
        request: RequestState,
    ) -> None:
        if not isinstance(request, RequestState):
            raise TypeError(
                f"RequestScheduler.add_request expects RequestState, got "
                f"{type(request).__name__}"
            )
        slot_idx = request.slot_idx
        self._requests[request_id] = request
        self._request_streams[request_id] = asyncio.Queue()
        if slot_idx is None:
            self._queued_ids.append(request_id)
            self._queued_ids_set.add(request_id)
            return
        slot_idx = int(slot_idx)
        request.slot_idx = slot_idx
        self._free_slots.remove(slot_idx)
        self._request_slots[request_id] = slot_idx
        self._add_running(request_id, request)

    def enqueue_request(
        self,
        request_id: str,
        request: RequestState,
    ) -> None:
        if not isinstance(request, RequestState):
            raise TypeError(
                f"RequestScheduler.enqueue_request expects RequestState, got "
                f"{type(request).__name__}"
            )
        request.slot_idx = None
        self.add_request(request_id, request)

    def admit_queued_requests(self, max_new: int | None = None) -> list[str]:
        admitted: list[str] = []
        admit_limit = len(self._free_slots) if max_new is None else max(0, int(max_new))
        while self._queued_ids and self._free_slots and len(admitted) < admit_limit:
            request_id = self._queued_ids.popleft()
            self._queued_ids_set.discard(request_id)
            request = self._requests[request_id]
            slot_idx = self.allocate_slot()
            request.slot_idx = slot_idx
            self._request_slots[request_id] = slot_idx
            self._add_running(request_id, request)
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
            if request_id not in self._queued_ids_set or not self._free_slots:
                continue
            self._queued_ids.remove(request_id)
            self._queued_ids_set.discard(request_id)
            request = self._requests[request_id]
            slot_idx = self.allocate_slot()
            request.slot_idx = slot_idx
            request.admitted_at = admitted_at
            self._request_slots[request_id] = slot_idx
            self._add_running(request_id, request)
            admitted.append(request_id)
        return admitted

    def reject_expired_queued_requests(
        self,
        *,
        now: float,
        max_queue_wait_s: float,
    ) -> list[tuple[str, str, RequestState]]:
        expired: list[tuple[str, str, RequestState]] = []
        if max_queue_wait_s <= 0:
            return expired
        for request_id in list(self._queued_ids):
            request = self._requests.get(request_id)
            if request is None:
                continue
            queued_at = float(request.queued_at or now)
            queue_wait = now - queued_at
            if queue_wait < max_queue_wait_s:
                continue
            self._queued_ids.remove(request_id)
            self._queued_ids_set.discard(request_id)
            removed_request = self._requests.pop(request_id, None)
            self._request_slots.pop(request_id, None)
            if removed_request is None:
                continue
            expired.append(
                (
                    request_id,
                    f"queue timeout after {queue_wait:.3f}s "
                    f"(limit={max_queue_wait_s:.3f}s)",
                    removed_request,
                )
            )
        return expired

    def transition_to_decode(self, request_id: str) -> None:
        """Move a running request from the prefill index to the decode index.

        Called by the execution backend when a prefill chunk finishes and the
        request starts decoding.
        """
        if request_id in self._running_ids_set:
            self._prefill_ids.discard(request_id)
            self._decode_ids.add(request_id)

    def get_request(self, request_id: str) -> RequestState:
        return self._requests[request_id]

    def stream_queue(
        self, request_id: str
    ) -> asyncio.Queue[RequestOutput | BaseException]:
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
        if queue is None:
            return
        if self._event_loop is not None and not self._is_event_loop_thread():
            self._event_loop.call_soon_threadsafe(queue.put_nowait, output)
            return
        queue.put_nowait(output)

    def publish_exception(self, request_id: str, exc: BaseException) -> None:
        queue = self._request_streams.get(request_id)
        if queue is None:
            return
        if self._event_loop is not None and not self._is_event_loop_thread():
            self._event_loop.call_soon_threadsafe(queue.put_nowait, exc)
            return
        queue.put_nowait(exc)

    def free_request(self, request_id: str) -> RequestState | None:
        request = self._requests.pop(request_id, None)
        if request is not None:
            slot_idx = request.slot_idx
            if slot_idx is not None:
                self._free_slots.append(int(slot_idx))
        self._request_slots.pop(request_id, None)
        self._remove_from_indexes(request_id)
        if request_id in self._running_ids:
            self._running_ids.remove(request_id)
        if request_id in self._queued_ids:
            self._queued_ids.remove(request_id)
        return request

    def abort_request(self, request_id: str) -> None:
        self.free_request(request_id)

    def classify_requests(self) -> tuple[list[str], list[str]]:
        # Preserve the admission order of running requests; ``set`` iteration is
        # not stable, so derive the lists from ``_running_ids`` while using the
        # indexes for O(1) membership checks.
        prefills = [rid for rid in self._running_ids if rid in self._prefill_ids]
        decodes = [rid for rid in self._running_ids if rid in self._decode_ids]
        return prefills, decodes

    def request_ids(self) -> list[str]:
        return list(self._running_ids) + list(self._queued_ids)
