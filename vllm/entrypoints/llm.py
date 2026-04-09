# SPDX-License-Identifier: Apache-2.0
import copy
import time
from typing import List, Optional

import torch

from vllm.engine.lite_engine import LiteEngine
from vllm.model_executor.layers.quantization.gguf import clear_gguf_cache
from vllm.model_executor.model_loader import get_tokenizer
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.serving.config_builder import build_vllm_config


class LLM:
    """Synchronous offline interface backed by ``LiteEngine``.

    This wrapper intentionally keeps a small API surface for lite-only offline
    generation and routes requests through the same runtime used by ``AsyncLLM``.
    """

    def __init__(self, model: str, **kwargs):
        self.model_path = model
        self.vllm_config = build_vllm_config(model, **kwargs)
        self.engine = LiteEngine(self.vllm_config)
        self.tokenizer = get_tokenizer(self.vllm_config.model_config)
        self.engine.tokenizer = self.tokenizer

    @property
    def model(self):
        return self.engine.model

    @torch.inference_mode()
    def generate(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None,
    ) -> List[RequestOutput]:
        if isinstance(prompts, str):
            prompts = [prompts]
        if sampling_params is None:
            sampling_params = SamplingParams()

        outputs: List[RequestOutput] = []
        batch_capacity = max(1, int(getattr(self.engine, "max_active_requests", 1)))

        for batch_start in range(0, len(prompts), batch_capacity):
            batch_prompts = prompts[batch_start : batch_start + batch_capacity]
            request_ids = [f"offline_{batch_start + i}_{time.time_ns()}" for i in range(len(batch_prompts))]
            latest_outputs: dict[str, RequestOutput] = {}

            for rid, prompt in zip(request_ids, batch_prompts):
                self.engine.add_request(rid, prompt, copy.deepcopy(sampling_params))

            unfinished = set(request_ids)
            while unfinished:
                step_outputs = self.engine.step()
                if not step_outputs:
                    raise RuntimeError(
                        "LiteEngine made no progress while requests were still active."
                    )
                for out in step_outputs:
                    latest_outputs[out.request_id] = out
                    if out.finished:
                        unfinished.discard(out.request_id)

            outputs.extend(latest_outputs[rid] for rid in request_ids)

        return outputs

    def shutdown(self) -> None:
        clear_gguf_cache()

    def stats(self) -> dict[str, object]:
        return self.engine.stats()

    def reset_stats(self, *, clear_prefix_cache: bool = False) -> None:
        self.engine.reset_stats(clear_prefix_cache=clear_prefix_cache)

    def register_lora_adapter(
        self,
        *,
        lora_name: str,
        lora_path: str | None = None,
        lora_int_id: int | None = None,
    ) -> dict[str, object]:
        return self.engine.register_lora_adapter(
            lora_name=lora_name,
            lora_path=lora_path,
            lora_int_id=lora_int_id,
        )

    def unregister_lora_adapter(self, lora_name: str) -> bool:
        return self.engine.unregister_lora_adapter(lora_name)
