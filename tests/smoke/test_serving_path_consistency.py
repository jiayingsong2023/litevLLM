# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib

from fastapi import FastAPI


def test_legacy_openai_app_object_is_shared() -> None:
    legacy_module = importlib.import_module("vllm.entrypoints.api_server")
    openai_module = importlib.import_module("vllm.entrypoints.openai.api_server")
    assert legacy_module.app is openai_module.app


def test_tokenize_and_pooling_serving_modules_importable() -> None:
    module_names = [
        "vllm.entrypoints.serve.tokenize.protocol",
        "vllm.entrypoints.serve.tokenize.serving",
        "vllm.entrypoints.serve.tokenize.api_router",
        "vllm.entrypoints.pooling.pooling.serving",
        "vllm.entrypoints.pooling.embed.serving",
        "vllm.entrypoints.pooling.score.serving",
        "vllm.entrypoints.pooling.classify.serving",
    ]
    for module_name in module_names:
        module = importlib.import_module(module_name)
        assert module is not None


def test_register_serve_router_does_not_fail() -> None:
    serve_module = importlib.import_module("vllm.entrypoints.serve")
    app = FastAPI()
    serve_module.register_vllm_serve_api_routers(app)
    assert len(app.router.routes) > 0


def test_async_utils_make_async_and_merge_iterators() -> None:
    async_utils = importlib.import_module("vllm.utils.async_utils")
    make_async = getattr(async_utils, "make_async")
    merge_async_iterators = getattr(async_utils, "merge_async_iterators")

    async def run_case() -> list[str]:
        wrapped = make_async(lambda x: x + 1)
        assert await wrapped(3) == 4

        async def stream(prefix: str):
            yield f"{prefix}-1"
            yield f"{prefix}-2"

        merged = []
        async for _, value in merge_async_iterators(stream("a"), stream("b")):
            merged.append(value)
        return merged

    values = asyncio.run(run_case())
    assert sorted(values) == ["a-1", "a-2", "b-1", "b-2"]
