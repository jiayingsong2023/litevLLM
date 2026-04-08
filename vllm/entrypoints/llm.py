# SPDX-License-Identifier: Apache-2.0
import copy
import json
import os
import time
from typing import List, Optional

import torch
from transformers import AutoConfig

from vllm.config import CacheConfig, LoadConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.engine.lite_engine import LiteEngine
from vllm.model_executor.layers.quantization.gguf import clear_gguf_cache
from vllm.model_executor.model_loader import get_tokenizer
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams


def _read_awq_group_size_and_bits(model_path: str) -> tuple[int, int]:
    group_size, bits = 128, 4
    config_path = os.path.join(model_path, "config.json")
    try:
        if not os.path.isfile(config_path):
            return group_size, bits
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = json.load(f)
        quantization_config = raw_config.get("quantization_config") or {}
        config_groups = quantization_config.get("config_groups")
        if isinstance(config_groups, dict):
            for group_value in config_groups.values():
                if not isinstance(group_value, dict):
                    continue
                weights_config = group_value.get("weights")
                if isinstance(weights_config, dict):
                    if weights_config.get("group_size") is not None:
                        group_size = int(weights_config["group_size"])
                    if weights_config.get("num_bits") is not None:
                        bits = int(weights_config["num_bits"])
                    break
        if quantization_config.get("group_size") is not None:
            group_size = int(quantization_config["group_size"])
        if quantization_config.get("bits") is not None:
            bits = int(quantization_config["bits"])
    except Exception:
        pass
    return group_size, bits


def _looks_like_awq_quantized_model(hf_config, model_path: str) -> bool:
    quantization_config = getattr(hf_config, "quantization_config", None)
    if isinstance(quantization_config, dict):
        quant_method = str(quantization_config.get("quant_method", "")).lower()
        if "awq" in quant_method:
            return True
        if quantization_config.get("format") == "pack-quantized":
            return True
    return os.path.isfile(os.path.join(model_path, "hf_quant_config.json"))


def _load_hf_config(model_path: str):
    try:
        return AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        class Simple:
            pass

        hf_config = Simple()
        hf_config.__dict__.update(data.get("text_config", data))
        hf_config.architectures = data.get("architectures", [])
        return hf_config


def _build_lite_vllm_config(model_path: str, **kwargs) -> VllmConfig:
    hf_config = _load_hf_config(model_path)
    max_model_len = int(kwargs.get("max_model_len", getattr(hf_config, "max_position_embeddings", 4096) or 4096))
    gpu_memory_utilization = float(kwargs.get("gpu_memory_utilization", 0.90))
    max_num_seqs = int(kwargs.get("max_num_seqs", kwargs.get("concurrent_reqs", 8)))
    max_num_batched_tokens = int(
        kwargs.get("max_num_batched_tokens", max(8192, max_num_seqs * min(max_model_len, 1024)))
    )

    model_config = ModelConfig(
        model=model_path,
        tokenizer=model_path,
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="float16",
        max_model_len=max_model_len,
    )
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
    )
    cache_config = CacheConfig(
        block_size=8,
        gpu_memory_utilization=gpu_memory_utilization,
        swap_space=0,
    )
    v_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        load_config=LoadConfig(load_format="auto"),
        quant_config=None,
    )

    if any(f.endswith(".gguf") for f in os.listdir(model_path)):
        from vllm.model_executor.layers.quantization.gguf import GGUFConfig

        v_config.quant_config = GGUFConfig()
    elif _looks_like_awq_quantized_model(hf_config, model_path):
        from vllm.model_executor.layers.quantization.awq import AWQConfig

        group_size, weight_bits = _read_awq_group_size_and_bits(model_path)
        v_config.quant_config = AWQConfig(weight_bits=weight_bits, group_size=group_size)
    return v_config


class LLM:
    """Synchronous offline interface backed by ``LiteEngine``.

    This wrapper intentionally keeps a small API surface for lite-only offline
    generation and routes requests through the same runtime used by ``AsyncLLM``.
    """

    def __init__(self, model: str, **kwargs):
        self.model_path = model
        self.vllm_config = _build_lite_vllm_config(model, **kwargs)
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
