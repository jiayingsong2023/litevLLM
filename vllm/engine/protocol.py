# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional, Union, Dict, Any
from vllm.sampling_params import SamplingParams
from vllm.outputs import RequestOutput

class EngineClient(ABC):
    """
    Standard interface for all LLM Engine frontends.
    """
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str,
        **kwargs,
    ) -> AsyncGenerator[RequestOutput, None]:
        pass

    @abstractmethod
    async def abort(self, request_ids: Union[str, List[str]]) -> None:
        pass

    @abstractmethod
    def shutdown(self) -> None:
        pass

    @abstractmethod
    async def get_model_config(self):
        pass
