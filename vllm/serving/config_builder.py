# SPDX-License-Identifier: Apache-2.0
import json
import os
from typing import Any

from transformers import AutoConfig

from vllm.config import CacheConfig, LoadConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.engine.runtime_config import RuntimeConfig
from vllm.transformers_utils.configs.gemma4 import build_fallback_hf_config


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


def _looks_like_awq_quantized_model(hf_config: Any, model_path: str) -> bool:
    quantization_config = getattr(hf_config, "quantization_config", None)
    if isinstance(quantization_config, dict):
        quant_method = str(quantization_config.get("quant_method", "")).lower()
        if "awq" in quant_method:
            return True
        if quantization_config.get("format") == "pack-quantized":
            return True
    return os.path.isfile(os.path.join(model_path, "hf_quant_config.json"))


def load_hf_config(model_path: str) -> Any:
    try:
        return AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return build_fallback_hf_config(data)


def build_vllm_config(model_path: str, **kwargs: Any) -> VllmConfig:
    hf_config = load_hf_config(model_path)
    max_model_len = int(
        kwargs.get(
            "max_model_len",
            getattr(hf_config, "max_position_embeddings", 4096) or 4096,
        )
    )
    gpu_memory_utilization = float(kwargs.get("gpu_memory_utilization", 0.90))
    max_num_seqs = int(kwargs.get("max_num_seqs", kwargs.get("concurrent_reqs", 8)))
    max_num_batched_tokens = int(
        kwargs.get("max_num_batched_tokens", max(8192, max_num_seqs * min(max_model_len, 1024)))
    )

    model_config = ModelConfig(
        model=model_path,
        tokenizer=kwargs.get("tokenizer", model_path),
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype=str(kwargs.get("dtype", "float16")),
        max_model_len=max_model_len,
    )
    model_config.hf_config = hf_config
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
    )
    cache_config = CacheConfig(
        block_size=int(kwargs.get("cache_block_size", 8)),
        gpu_memory_utilization=gpu_memory_utilization,
        swap_space=0,
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        load_config=LoadConfig(load_format="auto"),
        quant_config=None,
    )
    vllm_config.runtime_policy_mode = str(kwargs.get("policy_mode", "auto")).lower()

    if any(f.endswith(".gguf") for f in os.listdir(model_path)):
        from vllm.model_executor.layers.quantization.gguf import GGUFConfig

        vllm_config.quant_config = GGUFConfig()
    elif _looks_like_awq_quantized_model(hf_config, model_path):
        from vllm.model_executor.layers.quantization.awq import AWQConfig

        group_size, weight_bits = _read_awq_group_size_and_bits(model_path)
        vllm_config.quant_config = AWQConfig(weight_bits=weight_bits, group_size=group_size)

    runtime_config = RuntimeConfig.from_vllm_config(vllm_config)
    vllm_config.runtime_config = runtime_config
    return vllm_config
