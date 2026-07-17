# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field


class TokenizeRequest(BaseModel):
    prompt: str
    add_special_tokens: bool = True
    return_token_strs: bool = False


class DetokenizeRequest(BaseModel):
    tokens: list[int] = Field(default_factory=list)


def _tokenizer(request: Request) -> Any:
    """Resolve the tokenizer supplied by the application attaching this router."""
    tokenizer = getattr(request.app.state, "tokenizer", None)
    if tokenizer is None:
        engine = getattr(request.app.state, "engine", None)
        tokenizer = getattr(getattr(engine, "engine", None), "tokenizer", None)
    if tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="tokenize router requires app.state.tokenizer or app.state.engine",
        )
    return tokenizer


router = APIRouter()


@router.post("/tokenize")
async def tokenize(request: TokenizeRequest, raw_request: Request) -> dict[str, Any]:
    tokenizer = _tokenizer(raw_request)
    tokens = list(
        tokenizer.encode(
            request.prompt,
            add_special_tokens=request.add_special_tokens,
        )
    )
    token_strs = (
        list(tokenizer.convert_ids_to_tokens(tokens))
        if request.return_token_strs
        else None
    )
    return {
        "tokens": tokens,
        "token_strs": token_strs,
        "count": len(tokens),
        "max_model_len": int(getattr(tokenizer, "model_max_length", 0) or 0),
    }


@router.post("/detokenize")
async def detokenize(
    request: DetokenizeRequest,
    raw_request: Request,
) -> dict[str, str]:
    return {"prompt": _tokenizer(raw_request).decode(request.tokens)}


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
