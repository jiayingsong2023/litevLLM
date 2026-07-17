# SPDX-License-Identifier: Apache-2.0
"""Quantizers used by the maintained lite model loaders."""

from typing import Literal

from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

QuantizationMethods = Literal["awq", "compressed-tensors", "gguf"]
QUANTIZATION_METHODS = list(QuantizationMethods.__args__)


def get_quantization_config(quantization: str) -> type[QuantizationConfig]:
    if quantization == "awq":
        from .awq import AWQConfig

        return AWQConfig
    if quantization == "compressed-tensors":
        from .compressed_tensors.compressed_tensors import CompressedTensorsConfig

        return CompressedTensorsConfig
    if quantization == "gguf":
        from .gguf import GGUFConfig

        return GGUFConfig
    raise ImportError(
        f"Quantization method {quantization} is not available in this Lite build."
    )


__all__ = ["QuantizationConfig", "QuantizationMethods", "get_quantization_config"]
