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
    """Abstract base class for litevLLM executors (Single process only)."""

    uses_ray: bool = False
    supports_pp: bool = False

    @staticmethod
    def get_class(vllm_config: VllmConfig) -> type["Executor"]:
        from vllm.executor.uniproc_executor import UniProcExecutor
        return UniProcExecutor

    def __init__(self, vllm_config: VllmConfig) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self._init_executor()
        self.is_sleeping = False
        self.sleeping_tags: set[str] = set()
        self.kv_output_aggregator = None

    @abstractmethod
    def _init_executor(self) -> None:
        raise NotImplementedError

    def initialize_from_config(self, kv_cache_configs: list[KVCacheConfig]) -> None:
        self.collective_rpc("initialize_from_config", args=(kv_cache_configs,))
        self.collective_rpc("compile_or_warm_up_model")

    def register_failure_callback(self, callback: FailureCallback):
        pass

    def determine_available_memory(self) -> list[int]:
        return self.collective_rpc("determine_available_memory")

    def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
        return self.collective_rpc("get_kv_cache_spec")

    @abstractmethod
    def collective_rpc(self, method, timeout=None, args=(), kwargs=None, non_block: bool = False):
        raise NotImplementedError

    def get_kv_connector_handshake_metadata(self) -> list[dict[int, Any]]:
        return [{}]

    def execute_model(self, scheduler_output: SchedulerOutput, non_block: bool = False) -> ModelRunnerOutput | None:
        output = self.collective_rpc("execute_model", args=(scheduler_output,), non_block=non_block)
        return output[0]

    def sample_tokens(self, grammar_output: GrammarOutput | None, non_block: bool = False) -> ModelRunnerOutput:
        output = self.collective_rpc("sample_tokens", args=(grammar_output,), non_block=non_block)
        return output[0]

    def execute_dummy_batch(self) -> None:
        self.collective_rpc("execute_dummy_batch")

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        output: list[DraftTokenIds] = self.collective_rpc("take_draft_token_ids")
        return output[0]

    @property
    def max_concurrent_batches(self) -> int:
        return 1

    def profile(self, is_start: bool = True):
        self.collective_rpc("profile", args=(is_start,))

    def save_sharded_state(self, path: str, pattern: str | None = None, max_size: int | None = None) -> None:
        self.collective_rpc("save_sharded_state", kwargs=dict(path=path, pattern=pattern, max_size=max_size))

    @abstractmethod
    def check_health(self) -> None:
        raise NotImplementedError

    def shutdown(self) -> None:
        self.collective_rpc("shutdown")

    def init_kv_output_aggregator(self, connector: Any) -> None:
        pass

    @cached_property
    def supported_tasks(self) -> tuple[SupportedTask, ...]:
        output: list[tuple[SupportedTask, ...]] = self.collective_rpc("get_supported_tasks")
        return output[0]

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return all(self.collective_rpc("add_lora", args=(lora_request,)))

    def remove_lora(self, lora_id: int) -> bool:
        return all(self.collective_rpc("remove_lora", args=(lora_id,)))

    def pin_lora(self, lora_id: int) -> bool:
        return all(self.collective_rpc("pin_lora", args=(lora_id,)))

    def list_loras(self) -> set[int]:
        sets: list[set[int]] = self.collective_rpc("list_loras")
        return sets[0]

    def reset_mm_cache(self) -> None:
        self.collective_rpc("reset_mm_cache")

    def reset_encoder_cache(self) -> None:
        self.collective_rpc("reset_encoder_cache")

    def sleep(self, level: int = 1):
        if self.is_sleeping: return
        self.collective_rpc("sleep", kwargs=dict(level=level))
        self.sleeping_tags = {"weights", "kv_cache"}
        self.is_sleeping = True

    def wake_up(self, tags: list[str] | None = None):
        if not self.is_sleeping: return
        self.collective_rpc("wake_up", kwargs=dict(tags=tags))
        if tags:
            for tag in tags: self.sleeping_tags.discard(tag)
        else:
            self.sleeping_tags.clear()
        if not self.sleeping_tags: self.is_sleeping = False

    def reinitialize_distributed(self, reconfig_request: ReconfigureDistributedRequest) -> None:
        pass

from vllm.executor.uniproc_executor import UniProcExecutor as _UniProcExecutor
UniProcExecutor = _UniProcExecutor
