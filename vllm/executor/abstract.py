# SPDX-License-Identifier: Apache-2.0
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import Future
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.tasks import SupportedTask
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.engine.v1 import ReconfigureDistributedRequest
from vllm.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1_outputs import DraftTokenIds, ModelRunnerOutput
from vllm.worker.worker_base import WorkerBase

logger = init_logger(__name__)

_R = TypeVar("_R")
FailureCallback = Callable[[], None]

class Executor(ABC):
