# SPDX-License-Identifier: Apache-2.0
"""Keep the capability matrix's supported import surface importable."""

from importlib import import_module

import vllm


def test_supported_python_surface_is_exported() -> None:
    assert set(vllm.__all__) == {
        "LLM",
        "PoolingParams",
        "SamplingParams",
        "TextPrompt",
        "TokensPrompt",
        "clear_cache",
    }
    for name in vllm.__all__:
        assert hasattr(vllm, name)


def test_supported_service_modules_import() -> None:
    for module_name in (
        "vllm.engine.async_llm",
        "vllm.serving.config_builder",
        "vllm.entrypoints.api_server",
        "vllm.entrypoints.openai.api_server",
        "vllm.entrypoints.serve.tokenize",
    ):
        assert import_module(module_name)
