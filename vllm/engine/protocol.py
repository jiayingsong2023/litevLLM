# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams


class EngineClient(ABC):
    """
    Standard interface for all LLM Engine frontends.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str,
        **kwargs,
    ) -> AsyncGenerator[RequestOutput, None]:
        pass

    @abstractmethod
    async def abort(self, request_ids: str | list[str]) -> None:
        pass

    @abstractmethod
    def shutdown(self) -> None:
        pass

    @abstractmethod
    async def get_model_config(self):
        pass
