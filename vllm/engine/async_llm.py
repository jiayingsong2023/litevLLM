# SPDX-License-Identifier: Apache-2.0
from typing import AsyncGenerator, List, Optional, Union
from vllm.config import VllmConfig
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.engine.lite_engine import LiteEngine
from vllm.logger import init_logger

logger = init_logger(__name__)

class AsyncLLM:
    def __init__(self, vllm_config: VllmConfig, **kwargs):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.errored = False
        self.engine = LiteEngine(vllm_config)
        
    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig, **kwargs):
        return cls(vllm_config, **kwargs)
        
    async def generate(
        self,
        prompt: Union[str, dict],
        sampling_params: SamplingParams,
        request_id: str,
        **kwargs,
    ) -> AsyncGenerator[RequestOutput, None]:
        logger.debug(f"AsyncLLM: Starting request {request_id}")
        yield RequestOutput()
        
    async def abort(self, request_ids: List[str]):
        pass
        
    def shutdown(self):
        pass
