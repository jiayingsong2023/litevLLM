# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib


def test_lite_serving_entrypoints_import() -> None:
    modules = [
        "vllm.entrypoints.api_server",
        "vllm.entrypoints.openai.api_server",
        "vllm.entrypoints.serve.tokenize.api_router",
        "vllm.entrypoints.pooling",
    ]

    for module in modules:
        importlib.import_module(module)


def test_legacy_api_server_reexports_openai_app() -> None:
    legacy = importlib.import_module("vllm.entrypoints.api_server")
    openai = importlib.import_module("vllm.entrypoints.openai.api_server")

    assert legacy.app is openai.app
