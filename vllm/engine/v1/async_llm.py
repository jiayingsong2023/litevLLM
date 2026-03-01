# SPDX-License-Identifier: Apache-2.0
from typing import AsyncGenerator, List, Optional
from vllm.config import VllmConfig
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams

class AsyncLLM:
    def __init__(self, vllm_config: VllmConfig, **kwargs):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.errored = False
        
    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig, **kwargs):
        return cls(vllm_config, **kwargs)
        
    async def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str,
        **kwargs,
    ) -> AsyncGenerator[RequestOutput, None]:
        yield RequestOutput()
        
    async def abort(self, request_ids: List[str]):
        pass
        
    def shutdown(self):
        pass
