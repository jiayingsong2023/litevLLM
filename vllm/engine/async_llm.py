# SPDX-License-Identifier: Apache-2.0
from typing import AsyncGenerator, List, Optional, Union, Any
from vllm.config import VllmConfig
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.engine.async_driver import AsyncDriver
from vllm.engine.lite_engine import LiteEngine
from vllm.model_executor.model_loader import get_tokenizer
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

class AsyncLLM(EngineClient):
    """
    Asynchronous Frontend for LiteEngine.
    """
    def __init__(self, vllm_config: VllmConfig, **kwargs):
        self.vllm_config = vllm_config
        self.engine = LiteEngine(vllm_config)
        self.engine.tokenizer = get_tokenizer(vllm_config.model_config)
        self.driver = AsyncDriver(self.engine)

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig, **kwargs):
        return cls(vllm_config, **kwargs)

    async def get_model_config(self):
        return self.vllm_config.model_config

    async def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[Any] = None,
        **kwargs,
    ) -> AsyncGenerator[RequestOutput, None]:
        lora_id = getattr(lora_request, "lora_name", None) if lora_request else None
        self.engine.add_request(request_id, prompt, sampling_params, lora_id=lora_id)
        self.driver.notify_new_work()
        
        # Stream results back to API
        async for output in self.engine.get_request_stream(request_id):
            yield output

    async def abort(self, request_ids: Union[str, List[str]]):
        if isinstance(request_ids, str): request_ids = [request_ids]
        for rid in request_ids:
            self.engine.abort_request(rid)

    def shutdown(self):
        self.driver.shutdown()
