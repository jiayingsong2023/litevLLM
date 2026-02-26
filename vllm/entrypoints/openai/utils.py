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
