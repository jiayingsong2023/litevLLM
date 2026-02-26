# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING

from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1
    from vllm.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.engine.v1 import EngineCoreOutputs
    from vllm.kv_cache_interface import KVCacheConfig
    from vllm.metrics.stats import SchedulerStats
    from vllm.v1_outputs import DraftTokenIds, ModelRunnerOutput
    from vllm.request import Request, RequestStatus
    from vllm.structured_output import StructuredOutputManager

class SchedulerInterface(ABC):
    @abstractmethod
    def __init__(
        self,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
        structured_output_manager: "StructuredOutputManager",
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def schedule(self) -> "SchedulerOutput":
        raise NotImplementedError

    @abstractmethod
    def get_grammar_bitmask(
        self, scheduler_output: "SchedulerOutput"
    ) -> "GrammarOutput | None":
        raise NotImplementedError

    @abstractmethod
    def update_from_output(
        self,
        scheduler_output: "SchedulerOutput",
        model_runner_output: "ModelRunnerOutput",
    ) -> dict[int, "EngineCoreOutputs"]:
        raise NotImplementedError

    @abstractmethod
    def update_draft_token_ids(self, draft_token_ids: "DraftTokenIds") -> None:
        raise NotImplementedError

    @abstractmethod
    def update_draft_token_ids_in_output(
        self, draft_token_ids: "DraftTokenIds", scheduler_output: "SchedulerOutput"
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_request(self, request: "Request") -> None:
        raise NotImplementedError

    @abstractmethod
    def finish_requests(
        self,
        request_ids: str | Iterable[str],
        finished_status: "RequestStatus",
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_num_unfinished_requests(self) -> int:
        NOTE: This is different from `not self.has_unfinished_requests()`.

        The scheduler maintains an internal list of the requests finished in the
        previous step. This list is returned from the next call to schedule(),
        to be sent to the model runner in the next step to clear cached states
        for these finished requests.

        This method checks if this internal list of finished requests is
        non-empty. This information is useful for DP attention.

        This is particularly required when the model weights are live-updated.

        Args:
            reset_running_requests: If True, all the running requests will be
                preempted and moved to the waiting queue. Otherwise, this method
                will only reset the KV prefix cache when there is no running request
                taking KV cache.

        This should be called when model weights are updated to ensure
        stale vision embeddings are not reused.
        raise NotImplementedError

    @abstractmethod
    def make_stats(self) -> "SchedulerStats | None":
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
