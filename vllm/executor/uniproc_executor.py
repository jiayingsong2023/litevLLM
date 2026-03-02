# SPDX-License-Identifier: Apache-2.0
import os
from typing import Any, Callable, List, Optional, Union, Tuple
from concurrent.futures import Future

from vllm.logger import init_logger
from vllm.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.executor.abstract import Executor
from vllm.v1_outputs import AsyncModelRunnerOutput, DraftTokenIds, ModelRunnerOutput
from vllm.serial_utils import run_method

logger = init_logger(__name__)

class UniProcExecutor(Executor):
    """
    LitevLLM: Simplified Single-Process Executor.
    Optimized for single-GPU inference by removing all distributed overhead.
    """
    def _init_executor(self) -> Tuple[str, int, int]:
        # Return static local rank for single-GPU
        return "tcp://127.0.0.1:0", 0, 0

    def collective_rpc(
        self,
        method: Union[str, Callable],
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        non_block: bool = False,
        single_value: bool = False,
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        # Direct local execution without IPC or network overhead
        result = run_method(self.driver_worker, method, args, kwargs)
        
        if not non_block:
            return result if single_value else [result]

        # Async handling for non-blocking calls
        future = Future[Any]()
        future.set_result(result if single_value else [result])
        return future

    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> Union[ModelRunnerOutput, None, Future[Optional[ModelRunnerOutput]]]:
        return self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
            non_block=non_block,
            single_value=True,
        )

    def sample_tokens(
        self, grammar_output: Optional[GrammarOutput], non_block: bool = False
    ) -> Union[ModelRunnerOutput, None, Future[Optional[ModelRunnerOutput]]]:
        return self.collective_rpc(
            "sample_tokens",
            args=(grammar_output,),
            non_block=non_block,
            single_value=True,
        )

    def take_draft_token_ids(self) -> Optional[DraftTokenIds]:
        return self.collective_rpc("take_draft_token_ids", single_value=True)

    def check_health(self) -> None:
        """Always healthy in single-process mode."""
        return

    def shutdown(self) -> None:
        if self.driver_worker:
            self.driver_worker.shutdown()

class ExecutorWithExternalLauncher(UniProcExecutor):
    def _init_executor(self) -> Tuple[str, int, int]:
        return "tcp://127.0.0.1:0", 0, 0
