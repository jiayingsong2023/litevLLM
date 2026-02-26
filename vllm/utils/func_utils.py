# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import threading
import warnings
from collections.abc import Callable, Mapping
from functools import lru_cache, partial, wraps
from typing import Any, TypeVar

from typing_extensions import ParamSpec

from vllm.logger import init_logger

logger = init_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

def identity(value: T, **kwargs) -> T:
    disallows kwargs names that can also be positional arguments.
    Given a callable which has one or more keyword only params and a dict
    mapping param names to values, drop values that can be not be kwarg
    expanded to overwrite one or more keyword-only args. This is used in a
    few places to handle custom processor overrides for multimodal models,
    e.g., for profiling when processor options provided by the user
    may affect the number of mm tokens per instance.

    Args:
        callable: Callable which takes 0 or more keyword only arguments.
                  If None is provided, all overrides names are allowed.
        overrides: Potential overrides to be used when invoking the callable.
        allow_var_kwargs: Allows overrides that are expandable for var kwargs.

    Returns:
        Dictionary containing the kwargs to be leveraged which may be used
        to overwrite one or more keyword only arguments when invoking the
        callable.
