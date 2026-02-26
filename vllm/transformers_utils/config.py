# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Callable
from dataclasses import asdict
from functools import cache, partial
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, TypeAlias

import huggingface_hub
from huggingface_hub import get_safetensors_metadata
from packaging.version import Version
from transformers import GenerationConfig, PretrainedConfig
from transformers.models.auto.image_processing_auto import get_image_processor_config
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.models.auto.tokenization_auto import get_tokenizer_config
from transformers.utils import CONFIG_NAME as HF_CONFIG_NAME

from vllm import envs
from vllm.logger import init_logger
from vllm.transformers_utils.repo_utils import is_mistral_model_repo
from vllm.transformers_utils.utils import parse_safetensors_file_metadata

from .config_parser_base import ConfigParserBase
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

try:
    # Transformers v5
    from transformers.configuration_utils import ALLOWED_ATTENTION_LAYER_TYPES
except ImportError:
    # Transformers v4
    from transformers.configuration_utils import (
        ALLOWED_LAYER_TYPES as ALLOWED_ATTENTION_LAYER_TYPES,
    )

if envs.VLLM_USE_MODELSCOPE:
    from modelscope import AutoConfig
else:
    from transformers import AutoConfig

MISTRAL_CONFIG_NAME = "params.json"

logger = init_logger(__name__)

class LazyConfigDict(dict):
    def __getitem__(self, key):
        if isinstance(value := super().__getitem__(key), type):
            return value

        import vllm.transformers_utils.configs as configs

        return getattr(configs, value)

_CONFIG_REGISTRY: dict[str, type[PretrainedConfig]] = LazyConfigDict(
    afmoe="AfmoeConfig",
    bagel="BagelConfig",
    chatglm="ChatGLMConfig",
    deepseek_vl_v2="DeepseekVLV2Config",
    deepseek_v32="DeepseekV3Config",
    flex_olmo="FlexOlmoConfig",
    hunyuan_vl="HunYuanVLConfig",
    isaac="IsaacConfig",
    kimi_linear="KimiLinearConfig",
    kimi_vl="KimiVLConfig",
    kimi_k25="KimiK25Config",
    RefinedWeb="RWConfig",  # For tiiuae/falcon-40b(-instruct)
    RefinedWebModel="RWConfig",  # For tiiuae/falcon-7b(-instruct)
    jais="JAISConfig",
    mlp_speculator="MLPSpeculatorConfig",
    medusa="MedusaConfig",
    eagle="EAGLEConfig",
    speculators="SpeculatorsConfig",
    nemotron="NemotronConfig",
    olmo3="Olmo3Config",
    ovis="OvisConfig",
    step3_vl="Step3VLConfig",
    step3_text="Step3TextConfig",
    qwen3_next="Qwen3NextConfig",
    lfm2_moe="Lfm2MoeConfig",
    tarsier2="Tarsier2Config",
)

_CONFIG_ATTRS_MAPPING: dict[str, str] = {
    "llm_config": "text_config",
}

_AUTO_CONFIG_KWARGS_OVERRIDES: dict[str, dict[str, Any]] = {
    "internvl_chat": {"has_no_defaults_at_init": True},
    "Llama_Nemotron_Nano_VL": {"attn_implementation": "eager"},
    "NVLM_D": {"has_no_defaults_at_init": True},
}

def is_rope_parameters_nested(rope_parameters: dict[str, Any]) -> bool:
    if config_format not in _CONFIG_FORMAT_TO_CONFIG_PARSER:
        raise ValueError(f"Unknown config format `{config_format}`.")
    return _CONFIG_FORMAT_TO_CONFIG_PARSER[config_format]()

def register_config_parser(config_format: str):

    def _wrapper(config_parser_cls):
        if config_format in _CONFIG_FORMAT_TO_CONFIG_PARSER:
            logger.warning(
                "Config format `%s` is already registered, and will be "
                "overwritten by the new parser class `%s`.",
                config_format,
                config_parser_cls,
            )
        if not issubclass(config_parser_cls, ConfigParserBase):
            raise ValueError(
                "The config parser must be a subclass of `ConfigParserBase`."
            )
        _CONFIG_FORMAT_TO_CONFIG_PARSER[config_format] = config_parser_cls
        logger.info(
            "Registered config parser `%s` with config format `%s`",
            config_parser_cls,
            config_format,
        )
        return config_parser_cls

    return _wrapper

def set_default_rope_theta(config: PretrainedConfig, default_theta: float) -> None:
    if getattr(config, "rope_parameters", None) is None:
        config.rope_parameters = {"rope_type": "default"}
    if "rope_theta" not in config.rope_parameters:
        config.rope_parameters["rope_theta"] = default_theta

def patch_rope_parameters(config: PretrainedConfig) -> None:
    return (
        _uses_mrope(config)
        or _uses_mrope(config.get_text_config())
        or thinker_uses_mrope(config)
    )

def thinker_uses_mrope(config: PretrainedConfig) -> bool:
    xdrope_section = getattr(config, "xdrope_section", None)
    if xdrope_section is not None and isinstance(xdrope_section, list):
        return len(xdrope_section)
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is None:
        return 0

    if isinstance(rope_scaling, dict) and "xdrope_section" in rope_scaling:
        xdrope_section = rope_scaling["xdrope_section"]
        if xdrope_section is not None and isinstance(xdrope_section, list):
            return len(xdrope_section)

    return 0

def is_encoder_decoder(config: PretrainedConfig) -> bool:
    Detect if the model with this config is used with interleaved attention.
    Update kwargs for AutoConfig initialization based on model_type
    for old_attr, new_attr in _CONFIG_ATTRS_MAPPING.items():
        if hasattr(config, old_attr):
            if not hasattr(config, new_attr):
                config.update({new_attr: getattr(config, old_attr)})
            logger.debug("Remapped config attribute '%s' to '%s'", old_attr, new_attr)
    return config

def maybe_override_with_speculators(
    model: str,
    tokenizer: str | None,
    trust_remote_code: bool,
    revision: str | None = None,
    vllm_speculative_config: dict[str, Any] | None = None,
    **kwargs,
) -> tuple[str, str | None, dict[str, Any] | None]:
    if check_gguf_file(model):
        kwargs["gguf_file"] = Path(model).name
        gguf_model_repo = Path(model).parent
    elif is_remote_gguf(model):
        repo_id, _ = split_remote_gguf(model)
        gguf_model_repo = Path(repo_id)
    else:
        gguf_model_repo = None
    kwargs["local_files_only"] = huggingface_hub.constants.HF_HUB_OFFLINE
    config_dict, _ = PretrainedConfig.get_config_dict(
        model if gguf_model_repo is None else gguf_model_repo,
        revision=revision,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )
    speculators_config = config_dict.get("speculators_config")

    if speculators_config is None:
        # No speculators config found, return original values
        return model, tokenizer, vllm_speculative_config

    # Speculators format detected - process overrides
    from vllm.transformers_utils.configs.speculators.base import SpeculatorsConfig

    speculative_config = SpeculatorsConfig.extract_vllm_speculative_config(
        config_dict=config_dict
    )

    # Set the draft model to the speculators model
    speculative_config["model"] = model

    # Override model and tokenizer with the verifier model from config
    verifier_model = speculators_config["verifier"]["name_or_path"]
    model = tokenizer = verifier_model

    return model, tokenizer, speculative_config

def get_config(
    model: str | Path,
    trust_remote_code: bool,
    revision: str | None = None,
    code_revision: str | None = None,
    config_format: str | ConfigFormat = "auto",
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

    if config_format == "auto":
        try:
            # First check for Mistral to avoid defaulting to
            # Transformers implementation.
            if is_mistral_model_repo(
                model_name_or_path=str(model), revision=revision
            ) and file_or_path_exists(
                model=model, config_name=MISTRAL_CONFIG_NAME, revision=revision
            ):
                config_format = "mistral"
            elif (_is_gguf and not _is_remote_gguf) or file_or_path_exists(
                model, HF_CONFIG_NAME, revision=revision
            ):
                config_format = "hf"
            # Remote GGUF models must have config.json in repo,
            # otherwise the config can't be parsed correctly.
            # FIXME(Isotr0py): Support remote GGUF repos without config.json
            elif _is_remote_gguf and not file_or_path_exists(
                model, HF_CONFIG_NAME, revision=revision
            ):
                err_msg = (
                    "Could not find config.json for remote GGUF model repo. "
                    "To load remote GGUF model through `<repo_id>:<quant_type>`, "
                    "ensure your model has config.json (HF format) file. "
                    "Otherwise please specify --hf-config-path <original_repo> "
                    "in engine args to fetch config from unquantized hf model."
                )
                logger.error(err_msg)
                raise ValueError(err_msg)
            else:
                raise ValueError(
                    "Could not detect config format for no config file found. "
                    "With config_format 'auto', ensure your model has either "
                    "config.json (HF format) or params.json (Mistral format). "
                    "Otherwise please specify your_custom_config_format "
                    "in engine args for customized config parser."
                )

        except Exception as e:
            error_message = (
                "Invalid repository ID or local directory specified:"
                " '{model}'.\nPlease verify the following requirements:\n"
                "1. Provide a valid Hugging Face repository ID.\n"
                "2. Specify a local directory that contains a recognized "
                "configuration file.\n"
                "   - For Hugging Face models: ensure the presence of a "
                "'config.json'.\n"
                "   - For Mistral models: ensure the presence of a "
                "'params.json'.\n"
            ).format(model=model)

            raise ValueError(error_message) from e

    config_parser = get_config_parser(config_format)
    config_dict, config = config_parser.parse(
        model,
        trust_remote_code=trust_remote_code,
        revision=revision,
        code_revision=code_revision,
        hf_overrides=hf_overrides_kw,
        **kwargs,
    )

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

    # Exhaustively patch RoPE parameters everywhere they might be
    patch_rope_parameters(config)
    patch_rope_parameters(config.get_text_config())
    SubConfigs: TypeAlias = dict[str, PretrainedConfig]
    sub_configs: SubConfigs | None = getattr(config, "sub_configs", None)
    if sub_configs:
        for sub_config in sub_configs:
            patch_rope_parameters(getattr(config, sub_config))

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
    config_format: str | ConfigFormat = "auto",
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

def _download_mistral_config_file(model, revision) -> dict:
    config_file_name = "params.json"
    config_dict = get_hf_file_to_dict(config_file_name, model, revision)
    if config_dict is None:
        raise ValueError(
            f"Failed to load mistral '{config_file_name}' config for model "
            f"{model}. Please check if the model is a mistral-format model "
            f"and if the config file exists."
        )
    assert isinstance(config_dict, dict)
    return config_dict

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
