# SPDX-License-Identifier: Apache-2.0
import asyncio
import threading
from collections.abc import AsyncGenerator
from typing import Any

from vllm.config import VllmConfig
from vllm.engine.async_driver import AsyncDriver
from vllm.engine.lite_engine import LiteEngine
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_tokenizer
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams

logger = init_logger(__name__)


class AsyncLLM(EngineClient):
    """
    Asynchronous Frontend for LiteEngine.
    """

    def __init__(self, vllm_config: VllmConfig, **kwargs):
        self.vllm_config = vllm_config
        self.engine = LiteEngine(vllm_config)
        self._engine_lock = threading.RLock()
        self.engine._async_engine_lock = self._engine_lock
        self.engine.set_tokenizer(get_tokenizer(vllm_config.model_config))
        runtime_config = getattr(vllm_config, "runtime_config", None)
        min_step_interval_s = float(
            getattr(runtime_config, "async_driver_min_step_interval_s", 0.001)
        )
        self.driver = AsyncDriver(
            self.engine,
            min_step_interval_s=min_step_interval_s,
        )

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
        lora_request: Any | None = None,
        multi_modal_data: dict[str, Any] | None = None,
        **kwargs,
    ) -> AsyncGenerator[RequestOutput, None]:
        await self.submit(
            prompt,
            sampling_params,
            request_id,
            lora_request=lora_request,
            multi_modal_data=multi_modal_data,
            **kwargs,
        )
        async for output in self.stream(request_id):
            yield output

    async def submit(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Any | None = None,
        multi_modal_data: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Admit work before a caller starts consuming its output stream."""
        await asyncio.to_thread(
            self._submit_sync,
            prompt,
            sampling_params,
            request_id,
            lora_request,
            multi_modal_data,
            kwargs,
        )
        self.driver.notify_new_work()

    def _submit_sync(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Any | None,
        multi_modal_data: dict[str, Any] | None,
        kwargs: dict[str, Any],
    ) -> None:
        del kwargs
        lora_id = getattr(lora_request, "lora_name", None) if lora_request else None
        with self._engine_lock:
            try:
                self.engine.add_request(
                    request_id,
                    prompt,
                    sampling_params,
                    lora_id=lora_id,
                    lora_request=lora_request,
                    multi_modal_data=multi_modal_data,
                )
            except TypeError as exc:
                # Backward compatibility for engines/test doubles that don't yet
                # accept the multi_modal_data keyword argument.
                if "multi_modal_data" not in str(exc):
                    raise
                self.engine.add_request(
                    request_id,
                    prompt,
                    sampling_params,
                    lora_id=lora_id,
                    lora_request=lora_request,
                )

    async def stream(self, request_id: str) -> AsyncGenerator[RequestOutput, None]:
        finished = False
        try:
            async for output in self.engine.get_request_stream(request_id):
                finished = output.finished
                yield output
        finally:
            if not finished:
                await self.abort(request_id)

    async def close_stream(self, request_id: str) -> None:
        """Release an admitted stream that a caller will not iterate."""
        await self.abort(request_id)

    async def abort(self, request_ids: str | list[str]):
        await asyncio.to_thread(self._abort_sync, request_ids)

    def _abort_sync(self, request_ids: str | list[str]) -> None:
        if isinstance(request_ids, str):
            request_ids = [request_ids]
        with self._engine_lock:
            for rid in request_ids:
                self.engine.abort_request(rid)

    def stats(self) -> dict[str, Any]:
        with self._engine_lock:
            stats = dict(self.engine.stats())
            if hasattr(self.driver, "stats"):
                stats["async_driver"] = self.driver.stats()
            return stats

    def is_healthy(self) -> bool:
        with self._engine_lock:
            return getattr(self.engine, "_fatal_error", None) is None

    def register_lora_adapter(
        self,
        *,
        lora_name: str,
        lora_path: str | None = None,
        lora_int_id: int | None = None,
    ) -> dict[str, Any]:
        with self._engine_lock:
            return self.engine.register_lora_adapter(
                lora_name=lora_name,
                lora_path=lora_path,
                lora_int_id=lora_int_id,
            )

    def unregister_lora_adapter(self, lora_name: str) -> bool:
        with self._engine_lock:
            return self.engine.unregister_lora_adapter(lora_name)

    def reset_stats(self, *, clear_prefix_cache: bool = False) -> None:
        with self._engine_lock:
            self.engine.reset_stats(clear_prefix_cache=clear_prefix_cache)

    def shutdown(self):
        with self._engine_lock:
            scheduler = getattr(self.engine, "scheduler", None)
            if scheduler is not None:
                for request_id in scheduler.request_ids():
                    self.engine.abort_request(request_id)
        self.driver.shutdown()
