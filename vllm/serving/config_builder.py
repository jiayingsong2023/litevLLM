# SPDX-License-Identifier: Apache-2.0
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from transformers import AutoConfig

from vllm.config import (
    CacheConfig,
    LoadConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.engine.fastinference_config import resolve_fastinference_config
from vllm.engine.runtime_config import RuntimeConfig
from vllm.model_executor.models.deepseek_v4_flash.config import (
    DEEPSEEK_V4_FLASH_SHAPE,
    DeepSeekV4FlashMemoryPolicy,
)
from vllm.model_executor.models.deepseek_v4_flash.gguf_reader import (
    GGUFParseError,
    read_deepseek_v4_flash_gguf,
)
from vllm.transformers_utils.configs.gemma4 import build_fallback_hf_config


def _read_awq_group_size_and_bits(model_path: str) -> tuple[int, int]:
    group_size, bits = 128, 4
    config_path = os.path.join(model_path, "config.json")
    try:
        if not os.path.isfile(config_path):
            return group_size, bits
        with open(config_path, encoding="utf-8") as f:
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


def load_hf_config(
    model_path: str,
    *,
    trust_remote_code: bool = False,
    revision: str | None = None,
) -> Any:
    try:
        return AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
    except Exception:
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)
        return build_fallback_hf_config(data)


def _iter_local_gguf_paths(model_path: str) -> tuple[Path, ...]:
    path = Path(model_path)
    if path.is_file() and path.suffix.lower() == ".gguf":
        return (path,)
    if not path.is_dir():
        return ()
    return tuple(sorted(p for p in path.iterdir() if p.suffix.lower() == ".gguf"))


def _is_deepseek_v4_flash_hf_config(hf_config: Any) -> bool:
    model_type = str(getattr(hf_config, "model_type", "") or "").lower()
    archs = getattr(hf_config, "architectures", []) or []
    if model_type not in ("deepseek_v4", "deepseek4", "deepseek_v4_flash"):
        return False
    return any("deepseekv4flash" in str(arch).lower() for arch in archs)


def _is_deepseek_v4_flash_gguf(model_path: str) -> bool:
    for gguf_path in _iter_local_gguf_paths(model_path):
        try:
            metadata = read_deepseek_v4_flash_gguf(gguf_path).metadata
        except (GGUFParseError, OSError, ValueError):
            continue
        if str(metadata.get("general.architecture", "")).lower() == "deepseek4":
            return True
    return False


def _build_deepseek_v4_flash_hf_config() -> Any:
    return SimpleNamespace(
        model_type="deepseek_v4",
        architectures=["DeepSeekV4FlashForCausalLM"],
        max_position_embeddings=DeepSeekV4FlashMemoryPolicy.max_first_release_context,
        num_hidden_layers=DEEPSEEK_V4_FLASH_SHAPE.num_layers,
        num_attention_heads=DEEPSEEK_V4_FLASH_SHAPE.num_attention_heads,
        num_key_value_heads=DEEPSEEK_V4_FLASH_SHAPE.num_kv_heads,
        head_dim=DEEPSEEK_V4_FLASH_SHAPE.head_dim,
    )


def _validate_deepseek_v4_flash_context(
    hf_config: Any,
    max_model_len: int,
) -> int:
    if not _is_deepseek_v4_flash_hf_config(hf_config):
        return max_model_len
    return DeepSeekV4FlashMemoryPolicy().validate_context_length(max_model_len)


def build_vllm_config(model_path: str, **kwargs: Any) -> VllmConfig:
    fastinference_config = resolve_fastinference_config(
        config=kwargs.get("fastinference_config"),
        path=kwargs.get("fastinference_config_path"),
    )
    trust_remote_code = bool(
        kwargs.get("trust_remote_code", fastinference_config.model.trust_remote_code)
    )
    revision = kwargs.get("revision", fastinference_config.model.revision)
    if trust_remote_code and not revision:
        raise ValueError("model revision is required when trust_remote_code is enabled")
    hf_config = (
        _build_deepseek_v4_flash_hf_config()
        if _is_deepseek_v4_flash_gguf(model_path)
        else load_hf_config(
            model_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
    )
    max_model_len = int(
        kwargs.get(
            "max_model_len",
            getattr(hf_config, "max_position_embeddings", 4096) or 4096,
        )
    )
    max_model_len = _validate_deepseek_v4_flash_context(hf_config, max_model_len)
    gpu_memory_utilization = float(kwargs.get("gpu_memory_utilization", 0.90))
    max_num_seqs = int(kwargs.get("max_num_seqs", kwargs.get("concurrent_reqs", 8)))
    default_max_num_batched_tokens = max(
        8192,
        max_num_seqs * min(max_model_len, 1024),
    )
    max_num_batched_tokens = int(
        kwargs.get("max_num_batched_tokens", default_max_num_batched_tokens)
    )

    model_config = ModelConfig(
        model=model_path,
        tokenizer=kwargs.get("tokenizer", model_path),
        tokenizer_mode="auto",
        trust_remote_code=trust_remote_code,
        revision=revision,
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

    if _iter_local_gguf_paths(model_path):
        from vllm.model_executor.layers.quantization.gguf import GGUFConfig

        vllm_config.quant_config = GGUFConfig()
    elif _looks_like_awq_quantized_model(hf_config, model_path):
        from vllm.model_executor.layers.quantization.awq import AWQConfig

        group_size, weight_bits = _read_awq_group_size_and_bits(model_path)
        vllm_config.quant_config = AWQConfig(
            weight_bits=weight_bits,
            group_size=group_size,
        )

    vllm_config.fastinference_config = fastinference_config
    vllm_config.fastinference_config_path = kwargs.get("fastinference_config_path")
    runtime_config = RuntimeConfig.from_vllm_config(vllm_config)
    cache_config.cache_dtype = (
        "int4"
        if runtime_config.kv_cache_dtype in ("turbo_int4", "int4")
        else runtime_config.kv_cache_dtype
    )
    vllm_config.runtime_config = runtime_config
    return vllm_config
