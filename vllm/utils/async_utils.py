# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncGenerator, Callable
from concurrent.futures import Executor
from typing import Any, TypeVar

T = TypeVar("T")


class AsyncMicrobatchTokenizer:
    """Minimal async wrapper used by renderer implementations."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    async def encode(self, text: str, **kwargs: Any) -> list[int]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._tokenizer.encode(text, **kwargs),
        )


def make_async(func: Callable[..., T], executor: Executor | None = None):
    async def _runner(*args: Any, **kwargs: Any) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))

    return _runner


async def merge_async_iterators(
    *iterators: AsyncGenerator[Any, None],
) -> AsyncGenerator[tuple[int, Any], None]:
    iterator_map = {idx: iterator.__aiter__() for idx, iterator in enumerate(iterators)}
    tasks = {idx: asyncio.create_task(ait.__anext__()) for idx, ait in iterator_map.items()}

    while tasks:
        done, _ = await asyncio.wait(tasks.values(), return_when=asyncio.FIRST_COMPLETED)
        finished_ids: list[int] = []

        for task in done:
            iterator_index = next(
                idx for idx, pending_task in tasks.items() if pending_task is task
            )
            try:
                value = task.result()
                yield iterator_index, value
                tasks[iterator_index] = asyncio.create_task(
                    iterator_map[iterator_index].__anext__()
                )
            except StopAsyncIteration:
                finished_ids.append(iterator_index)

        for iterator_index in finished_ids:
            tasks.pop(iterator_index, None)
