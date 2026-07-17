# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Callable
from dataclasses import asdict
from functools import cache, partial
from pathlib import Path
from typing import Any

from huggingface_hub import get_safetensors_metadata
from transformers import AutoConfig, GenerationConfig, PretrainedConfig
from transformers.models.auto.image_processing_auto import get_image_processor_config
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.models.auto.tokenization_auto import get_tokenizer_config

from vllm import envs
from vllm.logger import init_logger
from vllm.transformers_utils.utils import parse_safetensors_file_metadata

from .gguf_utils import (
    check_gguf_file,
    is_gguf,
    is_remote_gguf,
    split_remote_gguf,
)
from .repo_utils import (
    file_or_path_exists,
    get_hf_file_to_dict,
    list_repo_files,
    try_get_local_file,
    with_retry,
)

logger = init_logger(__name__)


class LazyConfigDict(dict):
    def __getitem__(self, key):
        if isinstance(value := super().__getitem__(key), type):
            return value

        import vllm.transformers_utils.configs as configs

        return getattr(configs, value)


_CONFIG_REGISTRY: dict[str, type[PretrainedConfig]] = LazyConfigDict(
    gemma4="Gemma4Config",
)

_CONFIG_ATTRS_MAPPING: dict[str, str] = {
    "llm_config": "text_config",
}

_AUTO_CONFIG_KWARGS_OVERRIDES: dict[str, dict[str, Any]] = {
    "internvl_chat": {"has_no_defaults_at_init": True},
    "Llama_Nemotron_Nano_VL": {"attn_implementation": "eager"},
    "NVLM_D": {"has_no_defaults_at_init": True},
}


def get_config(
    model: str | Path,
    trust_remote_code: bool,
    revision: str | None = None,
    code_revision: str | None = None,
    config_format: str = "auto",
    hf_overrides_kw: dict[str, Any] | None = None,
    hf_overrides_fn: Callable[[PretrainedConfig], PretrainedConfig] | None = None,
    **kwargs,
) -> PretrainedConfig:
    # Separate model folder from file path for GGUF models

    _is_gguf = is_gguf(model)
    _is_remote_gguf = is_remote_gguf(model)
    if _is_gguf:
        if check_gguf_file(model):
            # Local GGUF file
            kwargs["gguf_file"] = Path(model).name
            model = Path(model).parent
        elif _is_remote_gguf:
            # Remote GGUF - extract repo_id from repo_id:quant_type format
            # The actual GGUF file will be downloaded later by GGUFModelLoader
            # Keep model as repo_id:quant_type for download, but use repo_id for config
            model, _ = split_remote_gguf(model)

    if config_format not in {"auto", "hf"}:
        raise ValueError("FastInference supports Hugging Face config.json only.")

    config_dict, _ = PretrainedConfig.get_config_dict(
        model,
        trust_remote_code=trust_remote_code,
        revision=revision,
        **kwargs,
    )
    config_cls = _CONFIG_REGISTRY.get(config_dict.get("model_type"))
    if config_cls is None:
        config = AutoConfig.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
            revision=revision,
            code_revision=code_revision,
            **kwargs,
        )
    else:
        config = config_cls.from_dict(config_dict)

    # Patching defaults for GGUF models
    if _is_gguf:
        # Some models have different default values between GGUF and HF.
        def apply_gguf_default(key: str, gguf_default: Any):
            if key not in config_dict:
                config.update({key: gguf_default})

        # Apply architecture-specific GGUF defaults.
        if config.model_type in {"qwen3_moe"}:
            # Qwen3 MoE: norm_topk_prob is always true.
            # Note that, this parameter is always false (HF default) on Qwen2 MoE.
            apply_gguf_default("norm_topk_prob", True)

    # Special architecture mapping check for GGUF models
    if _is_gguf:
        if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")
        model_type = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]
        config.update({"architectures": [model_type]})

    # Architecture mapping for models without explicit architectures field
    if not config.architectures:
        if config.model_type not in MODEL_MAPPING_NAMES:
            logger.warning(
                "Model config does not have a top-level 'architectures' field: "
                "expecting `hf_overrides={'architectures': ['...']}` to be passed "
                "in engine args."
            )
        else:
            model_type = MODEL_MAPPING_NAMES[config.model_type]
            config.update({"architectures": [model_type]})

    # ModelOpt 0.31.0 and after saves the quantization config in the model
    # config file.
    quantization_config = config_dict.get("quantization_config", None)

    # ModelOpt 0.29.0 and before saves the quantization config in a separate
    # "hf_quant_config.json" in the same directory as the model config file.
    if quantization_config is None and file_or_path_exists(
        model, "hf_quant_config.json", revision
    ):
        quantization_config = get_hf_file_to_dict(
            "hf_quant_config.json", model, revision
        )

    if quantization_config is not None:
        config.quantization_config = quantization_config
        # auto-enable DeepGEMM UE8M0 if model config requests it
        scale_fmt = quantization_config.get("scale_fmt", None)
        if scale_fmt in ("ue8m0",):
            if not envs.is_set("VLLM_USE_DEEP_GEMM_E8M0"):
                os.environ["VLLM_USE_DEEP_GEMM_E8M0"] = "1"
                logger.info_once(
                    (
                        "Detected quantization_config.scale_fmt=%s; "
                        "enabling UE8M0 for DeepGEMM."
                    ),
                    scale_fmt,
                )
            elif not envs.VLLM_USE_DEEP_GEMM_E8M0:
                logger.warning_once(
                    (
                        "Model config requests UE8M0 "
                        "(quantization_config.scale_fmt=%s), but "
                        "VLLM_USE_DEEP_GEMM_E8M0=0 is set; "
                        "UE8M0 for DeepGEMM disabled."
                    ),
                    scale_fmt,
                )

    if hf_overrides_kw:
        logger.debug("Overriding HF config with %s", hf_overrides_kw)
        config.update(hf_overrides_kw)
    if hf_overrides_fn:
        logger.debug("Overriding HF config with %s", hf_overrides_fn)
        config = hf_overrides_fn(config)

    if trust_remote_code:
        maybe_register_config_serialize_by_value()

    return config


@cache
def get_pooling_config(
    model: str,
    revision: str | None = "main",
) -> dict[str, Any] | None:
    if is_remote_gguf(model):
        model, _ = split_remote_gguf(model)

    modules_file_name = "modules.json"

    modules_dict = None
    if file_or_path_exists(
        model=model, config_name=modules_file_name, revision=revision
    ):
        modules_dict = get_hf_file_to_dict(modules_file_name, model, revision)

    if modules_dict is None:
        return None

    logger.info("Found sentence-transformers modules configuration.")

    pooling = next(
        (
            item
            for item in modules_dict
            if item["type"] == "sentence_transformers.models.Pooling"
        ),
        None,
    )
    normalize = bool(
        next(
            (
                item
                for item in modules_dict
                if item["type"] == "sentence_transformers.models.Normalize"
            ),
            False,
        )
    )

    if pooling:
        from vllm.config.pooler import SEQ_POOLING_TYPES, TOK_POOLING_TYPES

        pooling_file_name = "{}/config.json".format(pooling["path"])
        pooling_dict = get_hf_file_to_dict(pooling_file_name, model, revision) or {}

        logger.info("Found pooling configuration.")

        config: dict[str, Any] = {"use_activation": normalize}
        for key, val in pooling_dict.items():
            if val is True:
                pooling_type = parse_pooling_type(key)
                if pooling_type in SEQ_POOLING_TYPES:
                    config["seq_pooling_type"] = pooling_type
                elif pooling_type in TOK_POOLING_TYPES:
                    config["tok_pooling_type"] = pooling_type
                else:
                    logger.debug("Skipping unrelated field: %r=%r", key, val)

        return config

    return None


def parse_pooling_type(pooling_name: str):
    if "pooling_mode_" in pooling_name:
        pooling_name = pooling_name.replace("pooling_mode_", "")

    if "_" in pooling_name:
        pooling_name = pooling_name.split("_", 1)[0]

    if "lasttoken" in pooling_name:
        pooling_name = "last"

    return pooling_name.upper()


@cache
def get_sentence_transformer_tokenizer_config(
    model: str | Path, revision: str | None = "main"
) -> dict[str, Any] | None:
    sentence_transformer_config_files = [
        "sentence_bert_config.json",
        "sentence_roberta_config.json",
        "sentence_distilbert_config.json",
        "sentence_camembert_config.json",
        "sentence_albert_config.json",
        "sentence_xlm-roberta_config.json",
        "sentence_xlnet_config.json",
    ]
    encoder_dict = None

    for config_file in sentence_transformer_config_files:
        if (
            try_get_local_file(model=model, file_name=config_file, revision=revision)
            is not None
        ):
            encoder_dict = get_hf_file_to_dict(config_file, model, revision)
            if encoder_dict:
                break

    if not encoder_dict and not Path(model).is_absolute():
        try:
            # If model is on HuggingfaceHub, get the repo files
            repo_files = list_repo_files(model, revision=revision)
        except Exception:
            repo_files = []

        for config_name in sentence_transformer_config_files:
            if config_name in repo_files:
                encoder_dict = get_hf_file_to_dict(config_name, model, revision)
                if encoder_dict:
                    break

    if not encoder_dict:
        return None

    logger.info("Found sentence-transformers tokenize configuration.")

    if all(k in encoder_dict for k in ("max_seq_length", "do_lower_case")):
        return encoder_dict
    return None


def maybe_register_config_serialize_by_value() -> None:
    try:
        import transformers_modules

        transformers_modules_available = True
    except ImportError:
        transformers_modules_available = False

    try:
        import multiprocessing
        import pickle

        import cloudpickle

        from vllm.config import VllmConfig

        # Register multiprocessing reducers to handle cross-process
        # serialization of VllmConfig objects that may contain custom configs
        # from transformers_modules
        def _reduce_config(config: VllmConfig):
            return (pickle.loads, (cloudpickle.dumps(config),))

        multiprocessing.reducer.register(VllmConfig, _reduce_config)

        # Register transformers_modules with cloudpickle if available
        if transformers_modules_available:
            cloudpickle.register_pickle_by_value(transformers_modules)

            # Ray support removed in single-process mode.

    except Exception as e:
        logger.warning(
            "Unable to register remote classes used by"
            " trust_remote_code with by-value serialization. This may"
            " lead to a later error. If remote code is not needed"
            " remove `--trust-remote-code`",
            exc_info=e,
        )


def get_hf_image_processor_config(
    model: str | Path,
    hf_token: bool | str | None = None,
    revision: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    # ModelScope does not provide an interface for image_processor
    if envs.VLLM_USE_MODELSCOPE:
        return dict()
    # Separate model folder from file path for GGUF models
    if check_gguf_file(model):
        model = Path(model).parent
    elif is_remote_gguf(model):
        model, _ = split_remote_gguf(model)
    return get_image_processor_config(
        model, token=hf_token, revision=revision, **kwargs
    )


def get_hf_text_config(config: PretrainedConfig):
    text_config = config.get_text_config()

    if text_config is not config and not hasattr(text_config, "num_attention_heads"):
        raise ValueError(
            "The text_config extracted from the model config does not have "
            "`num_attention_heads` attribute. This indicates a mismatch "
            "between the model config and vLLM's expectations. Please "
            "ensure that the model config is compatible with vLLM."
        )

    return text_config


def try_get_generation_config(
    model: str,
    trust_remote_code: bool,
    revision: str | None = None,
    config_format: str = "auto",
) -> GenerationConfig | None:
    # GGUF files don't have generation_config.json - their config is embedded
    # in the file header. Skip all filesystem lookups to avoid re-reading the
    # memory-mapped file, which can hang in multi-process scenarios when the
    # EngineCore process already has the file mapped.
    if is_gguf(model):
        return None

    try:
        return GenerationConfig.from_pretrained(
            model,
            revision=revision,
        )
    except OSError:  # Not found
        try:
            config = get_config(
                model,
                trust_remote_code=trust_remote_code,
                revision=revision,
                config_format=config_format,
            )
            return GenerationConfig.from_model_config(config)
        except OSError:  # Not found
            return None


def try_get_safetensors_metadata(
    model: str,
    *,
    revision: str | None = None,
):
    get_safetensors_metadata_partial = partial(
        get_safetensors_metadata, model, revision=revision
    )

    try:
        return with_retry(
            get_safetensors_metadata_partial, "Error retrieving safetensors"
        )
    except Exception:
        return None


def try_get_tokenizer_config(
    pretrained_model_name_or_path: str | os.PathLike,
    trust_remote_code: bool,
    revision: str | None = None,
) -> dict[str, Any] | None:
    try:
        return get_tokenizer_config(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
    except Exception:
        return None


@cache
def try_get_dense_modules(
    model: str | Path,
    revision: str | None = None,
) -> list[dict[str, Any]] | None:
    try:
        modules = get_hf_file_to_dict("modules.json", model, revision)
        if not modules:
            return None

        if isinstance(modules, dict):
            modules = modules.get("modules", [])

        dense_modules = [
            m for m in modules if m.get("type") == "sentence_transformers.models.Dense"
        ]
        if not dense_modules:
            return None

        layer_configs = []
        for module in dense_modules:
            folder = module.get("path", "")

            config_path = f"{folder}/config.json" if folder else "config.json"
            layer_config = get_hf_file_to_dict(config_path, model, revision)
            if not layer_config:
                continue
            layer_config["folder"] = folder
            layer_configs.append(layer_config)
        return layer_configs
    except Exception:
        return None


def get_safetensors_params_metadata(
    model: str,
    *,
    revision: str | None = None,
) -> dict[str, Any]:
    full_metadata = {}
    if (model_path := Path(model)).exists():
        safetensors_to_check = model_path.glob("*.safetensors")
        full_metadata = {
            param_name: info
            for file_path in safetensors_to_check
            if file_path.is_file()
            for param_name, info in parse_safetensors_file_metadata(file_path).items()
        }
    else:
        repo_mt = try_get_safetensors_metadata(model, revision=revision)
        if repo_mt and (files_mt := repo_mt.files_metadata):
            full_metadata = {
                param_name: asdict(info)
                for file_mt in files_mt.values()
                for param_name, info in file_mt.tensors.items()
            }
    return full_metadata


def _maybe_retrieve_max_pos_from_hf(model, revision, **kwargs) -> int:
    max_position_embeddings = 128_000
    try:
        trust_remote_code_val = kwargs.get("trust_remote_code", False)
        hf_config = get_config(
            model=model,
            trust_remote_code=trust_remote_code_val,
            revision=revision,
            config_format="hf",
        )
        if hf_value := hf_config.get_text_config().max_position_embeddings:
            max_position_embeddings = hf_value
    except Exception as e:
        logger.warning(
            "The params.json file is missing 'max_position_embeddings'"
            " and could not get a value from the HF config."
            " Defaulting to 128000",
            exc_info=e,
        )

    return max_position_embeddings
