# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TypeVar

from fastapi import Request
from fastapi.exceptions import RequestValidationError

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
)

# Used internally
_ChatCompletionResponseChoiceT = TypeVar(
    "_ChatCompletionResponseChoiceT",
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
)

def maybe_filter_parallel_tool_calls(
    choice: _ChatCompletionResponseChoiceT, request: ChatCompletionRequest
) -> _ChatCompletionResponseChoiceT:
    del request
    return choice


async def validate_json_request(raw_request: Request) -> None:
    content_type = raw_request.headers.get("content-type", "")
    if "application/json" not in content_type:
        raise RequestValidationError(
            [
                {
                    "type": "value_error",
                    "loc": ("header", "content-type"),
                    "msg": "Content-Type must be application/json",
                    "input": content_type,
                }
            ]
        )
