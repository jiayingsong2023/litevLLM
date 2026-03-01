# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Iterable, Mapping
from typing import Any

from vllm.config import ModelConfig, VllmConfig
from vllm.inputs.data import PromptType, StreamingInput
from vllm.lora.request import LoRARequest
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.plugins.io_processors import IOProcessor
from vllm.pooling_params import PoolingParams
from vllm.renderers import BaseRenderer
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.engine.v1_protocol import EngineCoreRequest
from vllm.engine.v1.input_processor import InputProcessor

class EngineClient(ABC):
        ...

    @abstractmethod
    def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:

        Args:
            request_id: The unique id of the request,
                        or an iterable of such ids.
        ...

    @abstractmethod
    async def start_profile(self) -> None:
        ...

    @abstractmethod
    async def reset_mm_cache(self) -> None:
        ...

    @abstractmethod
    async def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        ...

    @abstractmethod
    async def wake_up(self, tags: list[str] | None = None) -> None:
        ...

    @abstractmethod
    async def add_lora(self, lora_request: LoRARequest) -> bool:

        Args:
            wait_for_inflight_requests: When ``True`` waits for in-flight requests
                to finish before pausing. When ``False`` (default), aborts in-flight
                requests immediately.
            clear_cache: Whether to clear KV and prefix caches after draining.
        ...

    @abstractmethod
    async def is_paused(self) -> bool:
        raise NotImplementedError

    async def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
    ):
        raise NotImplementedError
