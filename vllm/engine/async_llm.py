# SPDX-License-Identifier: Apache-2.0
import asyncio
from typing import AsyncGenerator, List, Optional, Union, Any
from vllm.config import VllmConfig
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
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
        
        self._background_loop_task: Optional[asyncio.Task] = None
        self._loop_running = False

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig, **kwargs):
        return cls(vllm_config, **kwargs)

    async def get_model_config(self):
        return self.vllm_config.model_config

    async def _engine_loop(self):
        """Background task that keeps the engine stepping."""
        while self._loop_running:
            try:
                # Run the engine step. 
                # Since this is a compute-bound task, we run it in the main loop
                # but ensure we yield back to the event loop immediately after.
                self.engine.step()
            except Exception as e:
                print(f"!!! AsyncLLM Loop Error: {e}")
            
            # Crucial: Allow the event loop to switch context
            # Use 0 to yield control without fixed delay
            await asyncio.sleep(0) 

    def _ensure_loop(self):
        if not self._loop_running:
            self._loop_running = True
            self._background_loop_task = asyncio.create_task(self._engine_loop())

    async def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[Any] = None,
        **kwargs,
    ) -> AsyncGenerator[RequestOutput, None]:
        self._ensure_loop()
        lora_id = getattr(lora_request, "lora_name", None) if lora_request else None
        self.engine.add_request(request_id, prompt, sampling_params, lora_id=lora_id)
        
        # Stream results back to API
        async for output in self.engine.get_request_stream(request_id):
            yield output

    async def abort(self, request_ids: Union[str, List[str]]):
        if isinstance(request_ids, str): request_ids = [request_ids]
        for rid in request_ids:
            if rid in self.engine._running_ids:
                self.engine._running_ids.remove(rid)

    def shutdown(self):
        self._loop_running = False
        if self._background_loop_task:
            self._background_loop_task.cancel()
