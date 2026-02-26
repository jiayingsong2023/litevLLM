# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import dataclasses
import functools
import os
from argparse import Namespace
from logging import Logger
from string import Template
from typing import TYPE_CHECKING

import regex as re
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask, BackgroundTasks

from vllm import envs
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import EmbedsPrompt, TokensPrompt
from vllm.inputs.parse import get_prompt_len
from vllm.logger import current_formatter_type, init_logger
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.completion.protocol import (
        CompletionRequest,
    )
    from vllm.entrypoints.openai.engine.protocol import (
        StreamOptions,
    )
    from vllm.entrypoints.openai.models.protocol import LoRAModulePath
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
else:
    ChatCompletionRequest = object
    CompletionRequest = object
    StreamOptions = object
    LoRAModulePath = object
    ResponsesRequest = object

logger = init_logger(__name__)

VLLM_SUBCMD_PARSER_EPILOG = (
    "For full list:            vllm {subcmd} --help=all\n"
    "For a section:            vllm {subcmd} --help=ModelConfig    (case-insensitive)\n"  # noqa: E501
    "For a flag:               vllm {subcmd} --help=max-model-len  (_ or - accepted)\n"  # noqa: E501
    "Documentation:            https://docs.vllm.ai\n"
)

async def listen_for_disconnect(request: Request) -> None:
    disconnections.

    This does _not_ use request.is_disconnected, which does not work with
    middleware. Instead this follows the pattern from
    starlette.StreamingResponse, which simultaneously awaits on two tasks- one
    to wait for an http disconnect message, and the other to do the work that we
    want done. When the first task finishes, the other is cancelled.

    A core assumption of this method is that the body of the request has already
    been read. This is a safe assumption to make for fastapi handlers that have
    already parsed the body of the request into a pydantic model for us.
    This decorator is unsafe to use elsewhere, as it will consume and throw away
    all incoming messages for the request while it looks for a disconnect
    message.

    In the case where a `StreamingResponse` is returned by the handler, this
    wrapper will stop listening for disconnects and instead the response object
    will start listening for disconnects.
