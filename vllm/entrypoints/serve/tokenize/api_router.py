# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from typing_extensions import assert_never

from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.serve.tokenize.protocol import (
    DetokenizeRequest,
    DetokenizeResponse,
    TokenizeRequest,
    TokenizeResponse,
)
from vllm.entrypoints.serve.tokenize.serving import OpenAIServingTokenization
from vllm.entrypoints.utils import with_cancellation


def tokenization(request: Request) -> OpenAIServingTokenization:
    return request.app.state.openai_serving_tokenization


router = APIRouter()


@router.post(
    "/tokenize",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
async def tokenize(request: TokenizeRequest, raw_request: Request):
    handler = tokenization(raw_request)
    result = await handler.create_tokenize(request, raw_request)

    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(), status_code=result.error.code)
    if isinstance(result, TokenizeResponse):
        return JSONResponse(content=result.model_dump())
    assert_never(result)


@router.post(
    "/detokenize",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
async def detokenize(request: DetokenizeRequest, raw_request: Request):
    handler = tokenization(raw_request)
    result = await handler.create_detokenize(request, raw_request)

    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(), status_code=result.error.code)
    if isinstance(result, DetokenizeResponse):
        return JSONResponse(content=result.model_dump())
    assert_never(result)


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
