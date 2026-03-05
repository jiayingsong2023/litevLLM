# SPDX-License-Identifier: Apache-2.0

from typing import Any

from pydantic import Field

from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel


class ChatCompletionToolsParam(OpenAIBaseModel):
    type: str = "function"
    function: dict[str, Any] | None = None


class ChatCompletionRequest(OpenAIBaseModel):
    model: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    parallel_tool_calls: bool | None = None


class ChatCompletionResponseChoice(OpenAIBaseModel):
    index: int = 0
    message: dict[str, Any] = Field(default_factory=dict)
    finish_reason: str | None = None


class ChatCompletionResponseStreamChoice(OpenAIBaseModel):
    index: int = 0
    delta: dict[str, Any] = Field(default_factory=dict)
    finish_reason: str | None = None
