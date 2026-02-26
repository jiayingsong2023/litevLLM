# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterator
from itertools import chain
from typing import TYPE_CHECKING

from vllm.sample.logits_processor.interface import (
    AddedRequest,
    BatchUpdate,
    MovedRequest,
    RemovedRequest,
)

if TYPE_CHECKING:
    from vllm.sample.logits_processor.interface import LogitsProcessor

class BatchUpdateBuilder:

    _removed: list[RemovedRequest]
    _is_removed_sorted: bool
    added: list[AddedRequest]
    moved: list[MovedRequest]

    def __init__(
        self,
        removed: list[RemovedRequest] | None = None,
        added: list[AddedRequest] | None = None,
        moved: list[MovedRequest] | None = None,
    ) -> None:
        self._removed = removed or []
        self.added = added or []
        self.moved = moved or []
        self._is_removed_sorted = False

        # Used to track changes in the pooling case
        # where we don't populate the added list.
        self.batch_changed = False

    def _ensure_removed_sorted(self) -> None:
        if not self._is_removed_sorted:
            self._removed.sort(reverse=True)
            self._is_removed_sorted = True

    @property
    def removed(self) -> list[RemovedRequest]:
        self._ensure_removed_sorted()
        return self._removed

    def removed_append(self, index: int) -> None:
        if self._is_removed_sorted:
            raise RuntimeError(
                "Cannot register new removed request after self.removed has been read."
            )
        self._removed.append(index)
        self.batch_changed = True

    def has_removed(self) -> bool:
        return bool(self._removed)

    def peek_removed(self) -> int | None:
        if self.has_removed():
            self._ensure_removed_sorted()
            return self._removed.pop()
        return None

    def reset(self) -> bool:
        internal batch update builder state.

        Args:
          batch_size: current persistent batch size

        Returns:
          Frozen logitsprocs batch update instance; `None` if no updates

    def __init__(self, logitsprocs: Iterator["LogitsProcessor"] | None = None) -> None:
        self.argmax_invariant: list[LogitsProcessor] = []
        self.non_argmax_invariant: list[LogitsProcessor] = []
        if logitsprocs:
            for logitproc in logitsprocs:
                (
                    self.argmax_invariant
                    if logitproc.is_argmax_invariant()
                    else self.non_argmax_invariant
                ).append(logitproc)

    @property
    def all(self) -> Iterator["LogitsProcessor"]:
