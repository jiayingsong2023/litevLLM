# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest


@pytest.mark.cpu_test
def test_legacy_api_server_imports() -> None:
    module = importlib.import_module("vllm.entrypoints.api_server")
    assert hasattr(module, "main")
    assert hasattr(module, "app")


@pytest.mark.cpu_test
def test_openai_serving_base_imports() -> None:
    module = importlib.import_module("vllm.entrypoints.openai.engine.serving")
    assert hasattr(module, "OpenAIServing")
    assert hasattr(module, "ServeContext")


@pytest.mark.cpu_test
def test_quantization_registry_guard_for_gptq() -> None:
    quant_module = importlib.import_module("vllm.model_executor.layers.quantization")
    get_config = getattr(quant_module, "get_quantization_config")

    with pytest.raises(ImportError, match="gptq is not available"):
        get_config("gptq")
