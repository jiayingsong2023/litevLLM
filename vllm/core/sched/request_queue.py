# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

from vllm.request import Request

class SchedulingPolicy(Enum):

    @abstractmethod
    def add_request(self, request: Request) -> None:
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        pass

    @abstractmethod
    def prepend_requests(self, requests: "RequestQueue") -> None:
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:

    def add_request(self, request: Request) -> None:
        return self.popleft()

    def peek_request(self) -> Request:
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        self.extendleft(requests)

    def remove_request(self, request: Request) -> None:
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
    A priority queue that supports heap operations.

    Respects the ordering defined in the Request class, where
    requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.
        heapq.heappush(self._heap, request)

    def pop_request(self) -> Request:
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0]

    def prepend_request(self, request: Request) -> None:
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        requests_to_remove = requests if isinstance(requests, set) else set(requests)
        self._heap = [r for r in self._heap if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
