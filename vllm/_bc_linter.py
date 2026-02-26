# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# vllm/_bc_linter.py
from collections.abc import Callable
from typing import Any, TypeVar, overload

T = TypeVar("T")

@overload
def bc_linter_skip(obj: T) -> T: ...

@overload
def bc_linter_skip(*, reason: str | None = ...) -> Callable[[T], T]: ...

def bc_linter_skip(obj: Any = None, *, reason: str | None = None):

    def _wrap(x: T) -> T:
        return x

    return _wrap if obj is None else obj

@overload
def bc_linter_include(obj: T) -> T: ...

@overload
def bc_linter_include(*, reason: str | None = ...) -> Callable[[T], T]: ...

def bc_linter_include(obj: Any = None, *, reason: str | None = None):

    def _wrap(x: T) -> T:
        return x

    return _wrap if obj is None else obj

__all__ = ["bc_linter_skip", "bc_linter_include"]
