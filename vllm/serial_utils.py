# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import importlib
import pickle
from collections.abc import Callable, Sequence
from functools import partial
from inspect import isclass
from types import FunctionType
from typing import Any, TypeAlias, get_type_hints

import cloudpickle
import msgspec
import numpy as np
import torch
import zmq
from msgspec import msgpack
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

from vllm import envs
from vllm.logger import init_logger
from vllm.multimodal.inputs import (
    BaseMultiModalField,
    MultiModalBatchedField,
    MultiModalFieldConfig,
    MultiModalFieldElem,
    MultiModalFlatField,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    MultiModalSharedField,
    NestedTensors,
)
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1_utils import tensor_data

logger = init_logger(__name__)

CUSTOM_TYPE_PICKLE = 1
CUSTOM_TYPE_CLOUDPICKLE = 2
CUSTOM_TYPE_RAW_VIEW = 3

# MultiModalField class serialization type map.
# These need to list all possible field types and match them
# to factory methods in `MultiModalFieldConfig`.
MMF_CLASS_TO_FACTORY: dict[type[BaseMultiModalField], str] = {
    MultiModalFlatField: "flat",
    MultiModalSharedField: "shared",
    MultiModalBatchedField: "batched",
}

bytestr: TypeAlias = bytes | bytearray | memoryview | zmq.Frame

def _log_insecure_serialization_warning():
    logger.warning_once(
        "Allowing insecure serialization using pickle due to "
        "VLLM_ALLOW_INSECURE_SERIALIZATION=1"
    )

def _typestr(val: Any) -> tuple[str, str] | None:
    if val is None:
        return None
    t = type(val)
    return t.__module__, t.__qualname__

def _encode_type_info_recursive(obj: Any) -> Any:
    if obj is None:
        return None
    if type(obj) is list:
        return [_encode_type_info_recursive(item) for item in obj]
    if type(obj) is dict:
        return {k: _encode_type_info_recursive(v) for k, v in obj.items()}
    return _typestr(obj)

def _decode_type_info_recursive(
    type_info: Any, data: Any, convert_fn: Callable[[Sequence[str], Any], Any]
) -> Any:
    if type_info is None:
        return data
    if isinstance(type_info, dict):
        assert isinstance(data, dict)
        return {
            k: _decode_type_info_recursive(type_info[k], data[k], convert_fn)
            for k in type_info
        }
    if isinstance(type_info, list) and (
        # Exclude serialized tensors/numpy arrays.
        len(type_info) != 2 or not isinstance(type_info[0], str)
    ):
        assert isinstance(data, list)
        return [
            _decode_type_info_recursive(ti, d, convert_fn)
            for ti, d in zip(type_info, data)
        ]
    return convert_fn(type_info, data)

class UtilityResult:

    Note that unlike vanilla `msgspec` Encoders, this interface is generally
    not thread-safe when encoding tensors / numpy arrays.

    By default, arrays below 256B are serialized inline Larger will get sent
    via dedicated messages. Note that this is a per-tensor limit.

    Note that unlike vanilla `msgspec` Decoders, this interface is generally
    not thread-safe when encoding tensors / numpy arrays.
    Run a method of an object with the given arguments and keyword arguments.
    If the method is string, it will be converted to a method using getattr.
    If the method is serialized bytes and will be deserialized using
    cloudpickle.
    If the method is a callable, it will be called directly.
        Make msgspec.Struct compatible with Pydantic, respecting defaults.
        Handle JSON=>msgspec.Struct. Used when exposing msgspec.Struct to the
        API as input or in `/docs`. Note this is cached by Pydantic and not
        called on every validation.
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        return msgspec.convert(value, type=cls)
