# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import contextlib
import multiprocessing
import time
import weakref
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from multiprocessing import connection
from multiprocessing.process import BaseProcess
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
    Union,
    overload,
)

import torch
from torch.autograd.profiler import record_function

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext, is_usage_stats_enabled, usage_message
from vllm.utils.network_utils import get_open_port, get_open_zmq_ipc_path, get_tcp_uri
from vllm.utils.system_utils import kill_process_tree
from vllm.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    import numpy as np

    from vllm.engine.v1.coordinator import DPCoordinator
    from vllm.engine.v1.utils import CoreEngineActorManager, CoreEngineProcManager

logger = init_logger(__name__)

T = TypeVar("T")

class ConstantList(Generic[T], Sequence):
    def __init__(self, x: list[T]) -> None:
        self._x = x

    def append(self, item):
        raise TypeError("Cannot append to a constant list")

    def extend(self, item):
        raise TypeError("Cannot extend a constant list")

    def insert(self, item):
        raise TypeError("Cannot insert into a constant list")

    def pop(self, item):
        raise TypeError("Cannot pop from a constant list")

    def remove(self, item):
        raise TypeError("Cannot remove from a constant list")

    def clear(self):
        raise TypeError("Cannot clear a constant list")

    def index(self, item: T, start: int = 0, stop: int | None = None) -> int:
        return self._x.index(item, start, stop if stop is not None else len(self._x))

    @overload
    def __getitem__(self, item: int) -> T: ...

    @overload
    def __getitem__(self, s: slice, /) -> list[T]: ...

    def __getitem__(self, item: int | slice) -> T | list[T]:
        return self._x[item]

    @overload
    def __setitem__(self, item: int, value: T): ...

    @overload
    def __setitem__(self, s: slice, value: T, /): ...

    def __setitem__(self, item: int | slice, value: T | list[T]):
        raise TypeError("Cannot set item in a constant list")

    def __delitem__(self, item):
        raise TypeError("Cannot delete item from a constant list")

    def __iter__(self):
        return iter(self._x)

    def __contains__(self, item):
        return item in self._x

    def __len__(self):
        return len(self._x)

    def __repr__(self):
        return f"ConstantList({self._x})"

    def copy(self) -> list[T]:
        return self._x.copy()

class CpuGpuBuffer:

    If local_only is True, participants are colocated and so a unique IPC
    address will be returned.

    Otherwise, the provided host and port will be used to construct a TCP

    Handles creation, monitoring, and termination of API server worker
    processes. Also monitors extra processes to check if they are healthy.

        Args:
            target_server_fn: Function to call for each API server process
            listen_address: Address to listen for client connections
            sock: Socket for client connections
            args: Command line arguments
            num_servers: Number of API server processes to start
            input_addresses: Input addresses for each API server
            output_addresses: Output addresses for each API server
            stats_update_address: Optional stats update address

    Raises an exception if any process exits with a non-zero status.

    Args:
        api_server_manager: The manager for API servers.
        engine_manager: The manager for engine processes.
            If CoreEngineProcManager, it manages local engines;
            if CoreEngineActorManager, it manages all engines.
        coordinator: The coordinator for data parallel.
    Copy the first length elements of a tensor into another tensor in a
    non-blocking manner.

    Used to copy pinned CPU tensor data to pre-allocated GPU tensors.

    Returns the sliced target tensor.

    if not is_usage_stats_enabled():
        return

    from vllm.model_executor.model_loader import get_architecture_class_name

    parallel_config = vllm_config.parallel_config

    # Prepare KV connector string if applicable
    kv_connector = None
    if vllm_config.kv_transfer_config is not None:
        kv_connector = vllm_config.kv_transfer_config.kv_connector

    usage_message.report_usage(
        get_architecture_class_name(vllm_config.model_config),
        usage_context,
        extra_kvs={
            # Common configuration
            "dtype": str(vllm_config.model_config.dtype),
            "block_size": vllm_config.cache_config.block_size,
            "gpu_memory_utilization": vllm_config.cache_config.gpu_memory_utilization,
            "kv_cache_memory_bytes": vllm_config.cache_config.kv_cache_memory_bytes,
            # Quantization
            "quantization": vllm_config.model_config.quantization,
            "kv_cache_dtype": str(vllm_config.cache_config.cache_dtype),
            # Feature flags
            "enable_lora": bool(vllm_config.lora_config),
            "enable_prefix_caching": vllm_config.cache_config.enable_prefix_caching,
            "enforce_eager": vllm_config.model_config.enforce_eager,
            "disable_custom_all_reduce": parallel_config.disable_custom_all_reduce,
            # Distributed parallelism settings
            "tensor_parallel_size": parallel_config.tensor_parallel_size,
            "data_parallel_size": parallel_config.data_parallel_size,
            "pipeline_parallel_size": parallel_config.pipeline_parallel_size,
            "enable_expert_parallel": parallel_config.enable_expert_parallel,
            # All2All backend for MoE expert parallel
            "all2all_backend": parallel_config.all2all_backend,
            # KV connector used
            "kv_connector": kv_connector,
        },
    )

_PROFILER_FUNC = None

def record_function_or_nullcontext(name: str) -> AbstractContextManager:
    global _PROFILER_FUNC

    # fast path assume it is set
    if _PROFILER_FUNC is not None:
        return _PROFILER_FUNC(name)

    func = contextlib.nullcontext
    if envs.VLLM_CUSTOM_SCOPES_FOR_PROFILING:
        func = record_function
    elif envs.VLLM_NVTX_SCOPES_FOR_PROFILING:
        import nvtx

        func = nvtx.annotate

    _PROFILER_FUNC = func
    return func(name)

def tensor_data(tensor: torch.Tensor) -> memoryview:
    return tensor.flatten().contiguous().view(torch.uint8).numpy().data

@dataclass
class IterationDetails:
    num_ctx_requests: int
    num_ctx_tokens: int
    num_generation_requests: int
    num_generation_tokens: int

    def __repr__(self) -> str:
        return f"IterationDetails(num_ctx_requests={self.num_ctx_requests},\
                 num_ctx_tokens={self.num_ctx_tokens}, \
                 num_generation_requests={self.num_generation_requests}, \
                 num_generation_tokens={self.num_generation_tokens})"

def compute_iteration_details(scheduler_output: SchedulerOutput) -> IterationDetails:
    num_context_requests = 0
    num_context_tokens = 0
    num_generation_requests = 0
    num_generation_tokens = 0
    new_req_ids = {new_req.req_id for new_req in scheduler_output.scheduled_new_reqs}
    for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
        if scheduler_output.scheduled_cached_reqs.is_context_phase(req_id) or (
            req_id in new_req_ids
        ):
            num_context_requests += 1
            num_context_tokens += num_tokens
        else:
            num_generation_requests += 1
            num_generation_tokens += num_tokens
    return IterationDetails(
        num_context_requests,
        num_context_tokens,
        num_generation_requests,
        num_generation_tokens,
    )
