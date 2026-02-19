# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal, get_args

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.platforms import current_platform

logger = init_logger(__name__)

QuantizationMethods = Literal[
    "awq",
    "fp8",
    "ptpc_fp8",
    "fbgemm_fp8",
    "fp_quant",
    "modelopt",
    "modelopt_fp4",
    "gguf",
    "gptq_marlin",
    "awq_marlin",
    "gptq",
    "compressed-tensors",
    "bitsandbytes",
    "experts_int8",
    "ipex",
    "quark",
    "moe_wna16",
    "torchao",
    "inc",
    "mxfp4",
    "petit_nvfp4",
    "cpu_awq",
]
QUANTIZATION_METHODS: list[str] = list(get_args(QuantizationMethods))

DEPRECATED_QUANTIZATION_METHODS = [
    "tpu_int8",
    "ptpc_fp8",
    "fbgemm_fp8",
    "fp_quant",
    "experts_int8",
    "ipex",
    "petit_nvfp4",
]

# The customized quantization methods which will be added to this dict.
_CUSTOMIZED_METHOD_TO_QUANT_CONFIG = {}


def register_quantization_config(quantization: str):
    """Register a customized vllm quantization config.

    When a quantization method is not supported by vllm, you can register a customized
    quantization config to support it.

    Args:
        quantization (str): The quantization method name.

    Examples:
        >>> from vllm.model_executor.layers.quantization import (
        ...     register_quantization_config,
        ... )
        >>> from vllm.model_executor.layers.quantization import get_quantization_config
        >>> from vllm.model_executor.layers.quantization.base_config import (
        ...     QuantizationConfig,
        ... )
        >>>
        >>> @register_quantization_config("my_quant")
        ... class MyQuantConfig(QuantizationConfig):
        ...     pass
        >>>
        >>> get_quantization_config("my_quant")
        <class 'MyQuantConfig'>
    """  # noqa: E501

    def _wrapper(quant_config_cls):
        if quantization in QUANTIZATION_METHODS:
            logger.warning(
                "The quantization method '%s' already exists and will be "
                "overwritten by the quantization config %s.",
                quantization,
                quant_config_cls,
            )
        else:
            QUANTIZATION_METHODS.append(quantization)
            # Automatically assume the custom quantization config is supported
            if sq := current_platform.supported_quantization:
                sq.append(quantization)

        if not issubclass(quant_config_cls, QuantizationConfig):
            raise ValueError(
                "The quantization config must be a subclass of `QuantizationConfig`."
            )
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization] = quant_config_cls
        return quant_config_cls

    return _wrapper


def get_quantization_config(quantization: str) -> type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")

    # lazy import to avoid triggering `torch.compile` too early
    method_to_config: dict[str, type[QuantizationConfig]] = {}
    
    try:
        from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
        method_to_config["quark"] = QuarkConfig
    except ImportError: pass

    try:
        from .awq import AWQConfig
        method_to_config["awq"] = AWQConfig
    except ImportError: pass

    try:
        from .awq_marlin import AWQMarlinConfig
        method_to_config["awq_marlin"] = AWQMarlinConfig
    except ImportError: pass

    try:
        from .bitsandbytes import BitsAndBytesConfig
        method_to_config["bitsandbytes"] = BitsAndBytesConfig
    except ImportError: pass

    try:
        from .compressed_tensors.compressed_tensors import CompressedTensorsConfig
        method_to_config["compressed-tensors"] = CompressedTensorsConfig
    except ImportError: pass

    try:
        from .cpu_wna16 import CPUAWQConfig
        method_to_config["cpu_awq"] = CPUAWQConfig
    except ImportError: pass

    try:
        from .experts_int8 import ExpertsInt8Config
        method_to_config["experts_int8"] = ExpertsInt8Config
    except ImportError: pass

    try:
        from .fbgemm_fp8 import FBGEMMFp8Config
        method_to_config["fbgemm_fp8"] = FBGEMMFp8Config
    except ImportError: pass

    try:
        from .fp8 import Fp8Config
        method_to_config["fp8"] = Fp8Config
    except ImportError: pass

    try:
        from .fp_quant import FPQuantConfig
        method_to_config["fp_quant"] = FPQuantConfig
    except ImportError: pass

    try:
        from .gguf import GGUFConfig
        method_to_config["gguf"] = GGUFConfig
    except ImportError: pass

    try:
        from .gptq import GPTQConfig
        method_to_config["gptq"] = GPTQConfig
    except ImportError: pass

    try:
        from .gptq_marlin import GPTQMarlinConfig
        method_to_config["gptq_marlin"] = GPTQMarlinConfig
    except ImportError: pass

    try:
        from .inc import INCConfig
        method_to_config["inc"] = INCConfig
        method_to_config["auto-round"] = INCConfig
    except ImportError: pass

    try:
        from .ipex_quant import IPEXConfig
        method_to_config["ipex"] = IPEXConfig
    except ImportError: pass

    try:
        from .modelopt import ModelOptFp8Config, ModelOptNvFp4Config
        method_to_config["modelopt"] = ModelOptFp8Config
        method_to_config["modelopt_fp4"] = ModelOptNvFp4Config
    except ImportError: pass

    try:
        from .moe_wna16 import MoeWNA16Config
        method_to_config["moe_wna16"] = MoeWNA16Config
    except ImportError: pass

    try:
        from .mxfp4 import Mxfp4Config
        method_to_config["mxfp4"] = Mxfp4Config
    except ImportError: pass

    try:
        from .petit import PetitNvFp4Config
        method_to_config["petit_nvfp4"] = PetitNvFp4Config
    except ImportError: pass

    try:
        from .ptpc_fp8 import PTPCFp8Config
        method_to_config["ptpc_fp8"] = PTPCFp8Config
    except ImportError: pass

    try:
        from .torchao import TorchAOConfig
        method_to_config["torchao"] = TorchAOConfig
    except ImportError: pass

    # Update the `method_to_config` with customized quantization methods.
    method_to_config.update(_CUSTOMIZED_METHOD_TO_QUANT_CONFIG)

    if quantization not in method_to_config:
        raise ImportError(f"Quantization method {quantization} is not available in this simplified build.")

    return method_to_config[quantization]


__all__ = [
    "QuantizationConfig",
    "QuantizationMethods",
    "get_quantization_config",
    "register_quantization_config",
    "QUANTIZATION_METHODS",
]
