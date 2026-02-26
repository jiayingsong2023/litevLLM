# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import cached_property
from multiprocessing import Lock
from typing import Any

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.network_utils import get_distributed_init_method, get_ip, get_open_port
from vllm.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.engine.v1 import ReconfigureDistributedRequest, ReconfigureRankType
from vllm.executor.abstract import Executor
from vllm.v1_outputs import AsyncModelRunnerOutput, DraftTokenIds, ModelRunnerOutput
from vllm.serial_utils import run_method
from vllm.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)

class UniProcExecutor(Executor):
    def _init_executor(self) -> None:
        distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())
        # set local rank as the device index if specified
        device_info = self.vllm_config.device_config.device.__str__().split(":")
        local_rank = int(device_info[1]) if len(device_info) > 1 else 0
        return distributed_init_method, 0, local_rank

    @cached_property
    def max_concurrent_batches(self) -> int:
        return 2 if self.scheduler_config.async_scheduling else 1

    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        single_value: bool = False,
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        if not non_block:
            result = run_method(self.driver_worker, method, args, kwargs)
            return result if single_value else [result]

        try:
            result = run_method(self.driver_worker, method, args, kwargs)
            if isinstance(result, AsyncModelRunnerOutput):
                if (async_thread := self.async_output_thread) is not None:
                    if single_value:
                        return async_thread.submit(result.get_output)

                    def get_output_list() -> list[Any]:
                        return [result.get_output()]

                    return async_thread.submit(get_output_list)
                result = result.get_output()
            future = Future[Any]()
            future.set_result(result if single_value else [result])
        except Exception as e:
            future = Future[Any]()
            future.set_exception(e)
        return future

    def execute_model(  # type: ignore[override]
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        return self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
            non_block=non_block,
            single_value=True,
        )

    def sample_tokens(  # type: ignore[override]
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        return self.collective_rpc(
            "sample_tokens",
            args=(grammar_output,),
            non_block=non_block,
            single_value=True,
        )

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.collective_rpc("take_draft_token_ids", single_value=True)

    def check_health(self) -> None:
        # UniProcExecutor will always be healthy as long as
        # it's running.
        return

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        self.driver_worker.reinitialize_distributed(reconfig_request)
        if (
            reconfig_request.new_data_parallel_rank
            == ReconfigureRankType.SHUTDOWN_CURRENT_RANK
        ):
            self.shutdown()

    def shutdown(self) -> None:
        if worker := self.driver_worker:
            worker.shutdown()

class ExecutorWithExternalLauncher(UniProcExecutor):

    def _init_executor(self) -> None:
