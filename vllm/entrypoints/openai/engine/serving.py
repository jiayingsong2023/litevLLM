# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import json
import sys
import time
import traceback
from collections.abc import AsyncGenerator, Callable, Mapping
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any, ClassVar, Generic, Protocol, TypeAlias, TypeVar

import numpy as np
from fastapi import Request
from openai.types.responses import (
    ToolChoiceFunction,
)
from pydantic import ConfigDict, TypeAdapter
from starlette.datastructures import Headers

import vllm.envs as envs
from vllm.beam_search import BeamSearchSequence, create_sort_beams_key_function
from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateContentFormatOption,
    ConversationMessage,
)
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponse,
)
from vllm.entrypoints.openai.engine.protocol import (
    ErrorInfo,
    ErrorResponse,
    FunctionCall,
    FunctionDefinition,
)
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.responses.context import (
    ConversationContext,
    HarmonyContext,
    ParsableContext,
    StreamingHarmonyContext,
)
from vllm.entrypoints.openai.responses.protocol import (
    ResponseInputOutputItem,
    ResponsesRequest,
)
from vllm.entrypoints.openai.responses.utils import (
    construct_input_messages,
)
from vllm.entrypoints.pooling.classify.protocol import (
    ClassificationChatRequest,
    ClassificationCompletionRequest,
    ClassificationResponse,
)
from vllm.entrypoints.pooling.embed.protocol import (
    EmbeddingBytesResponse,
    EmbeddingChatRequest,
    EmbeddingCompletionRequest,
    EmbeddingResponse,
)
from vllm.entrypoints.pooling.pooling.protocol import (
    IOProcessorRequest,
    PoolingChatRequest,
    PoolingCompletionRequest,
    PoolingResponse,
)
from vllm.entrypoints.pooling.score.protocol import (
    RerankRequest,
    ScoreDataRequest,
    ScoreQueriesDocumentsRequest,
    ScoreRequest,
    ScoreResponse,
    ScoreTextRequest,
)
from vllm.entrypoints.serve.tokenize.protocol import (
    DetokenizeRequest,
    TokenizeChatRequest,
    TokenizeCompletionRequest,
    TokenizeResponse,
)
from vllm.entrypoints.utils import get_max_tokens, sanitize_message
from vllm.exceptions import VLLMValidationError
from vllm.inputs.data import EmbedsPrompt, PromptType, TokensPrompt
from vllm.inputs.parse import (
    get_prompt_components,
    is_explicit_encoder_decoder_prompt,
)
from vllm.logger import init_logger
from vllm.logprobs import Logprob, PromptLogprobs
from vllm.lora.request import LoRARequest
from vllm.multimodal import MultiModalDataDict
from vllm.outputs import CompletionOutput, PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.renderers import ChatParams, TokenizeParams, merge_kwargs
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser, ToolParserManager
from vllm.tracing import (
    contains_trace_headers,
    extract_trace_headers,
    log_tracing_disabled_warning,
)
from vllm.utils import random_uuid
from vllm.utils.async_utils import (
    collect_from_async_generator,
    merge_async_iterators,
)

class GenerationError(Exception):
    A short string prepended to every request’s ID (e.g. "embd", "classify")
    so you can easily tell “this ID came from Embedding vs Classification.”
        parser = None
        if not enable_auto_tools or tool_parser_name is None:
            return parser
        logger.info('"auto" tool choice has been enabled.')

        try:
            if tool_parser_name == "pythonic" and self.model_config.model.startswith(
                "meta-llama/Llama-3.2"
            ):
                logger.warning(
                    "Llama3.2 models may struggle to emit valid pythonic tool calls"
                )
            parser = ToolParserManager.get_tool_parser(tool_parser_name)
        except Exception as e:
            raise TypeError(
                "Error: --enable-auto-tool-choice requires "
                f"tool_parser:'{tool_parser_name}' which has not "
                "been registered"
            ) from e
        return parser

    def _get_reasoning_parser(
        self,
        reasoning_parser_name: str,
    ) -> Callable[[TokenizerLike], ReasoningParser] | None:
        Default preprocessing hook. Subclasses may override
        to prepare `ctx` (classification, embedding, etc.).
        Default response builder. Subclass may override this method
        to return the appropriate response object.
        if error := await self._check_model(ctx.request):
            yield error
        if error := self._validate_request(ctx):
            yield error

        preprocess_ret = await self._preprocess(ctx)
        if isinstance(preprocess_ret, ErrorResponse):
            yield preprocess_ret

        generators_ret = await self._prepare_generators(ctx)
        if isinstance(generators_ret, ErrorResponse):
            yield generators_ret

        collect_ret = await self._collect_batch(ctx)
        if isinstance(collect_ret, ErrorResponse):
            yield collect_ret

        yield self._build_response(ctx)

    def _validate_request(self, ctx: ServeContext) -> ErrorResponse | None:
        truncate_prompt_tokens = getattr(ctx.request, "truncate_prompt_tokens", None)

        if (
            truncate_prompt_tokens is not None
            and truncate_prompt_tokens > self.max_model_len
        ):
            return self.create_error_response(
                "truncate_prompt_tokens value is "
                "greater than max_model_len."
                " Please, select a smaller truncation size."
            )
        return None

    def _create_pooling_params(
        self,
        ctx: ServeContext,
    ) -> PoolingParams | ErrorResponse:
        if not hasattr(ctx.request, "to_pooling_params"):
            return self.create_error_response(
                "Request type does not support pooling parameters"
            )

        return ctx.request.to_pooling_params()

    async def _prepare_generators(
        self,
        ctx: ServeContext,
    ) -> ErrorResponse | None:
        try:
            if ctx.engine_prompts is None:
                return self.create_error_response("Engine prompts not available")

            num_prompts = len(ctx.engine_prompts)
            final_res_batch: list[PoolingRequestOutput | None]
            final_res_batch = [None] * num_prompts

            if ctx.result_generator is None:
                return self.create_error_response("Result generator not available")

            async for i, res in ctx.result_generator:
                final_res_batch[i] = res

            if None in final_res_batch:
                return self.create_error_response(
                    "Failed to generate results for all prompts"
                )

            ctx.final_res_batch = [res for res in final_res_batch if res is not None]

            return None

        except Exception as e:
            return self.create_error_response(e)

    def create_error_response(
        self,
        message: str | Exception,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
        param: str | None = None,
    ) -> ErrorResponse:
        exc: Exception | None = None

        if isinstance(message, Exception):
            exc = message

            from vllm.exceptions import VLLMValidationError

            if isinstance(exc, VLLMValidationError):
                err_type = "BadRequestError"
                status_code = HTTPStatus.BAD_REQUEST
                param = exc.parameter
            elif isinstance(exc, (ValueError, TypeError, RuntimeError, OverflowError)):
                # Common validation errors from user input
                err_type = "BadRequestError"
                status_code = HTTPStatus.BAD_REQUEST
                param = None
            elif isinstance(exc, NotImplementedError):
                err_type = "NotImplementedError"
                status_code = HTTPStatus.NOT_IMPLEMENTED
                param = None
            elif exc.__class__.__name__ == "TemplateError":
                # jinja2.TemplateError (avoid importing jinja2)
                err_type = "BadRequestError"
                status_code = HTTPStatus.BAD_REQUEST
                param = None
            else:
                err_type = "InternalServerError"
                status_code = HTTPStatus.INTERNAL_SERVER_ERROR
                param = None

            message = str(exc)

        if self.log_error_stack:
            exc_type, _, _ = sys.exc_info()
            if exc_type is not None:
                traceback.print_exc()
            else:
                traceback.print_stack()

        return ErrorResponse(
            error=ErrorInfo(
                message=sanitize_message(message),
                type=err_type,
                code=status_code.value,
                param=param,
            )
        )

    def create_streaming_error_response(
        self,
        message: str | Exception,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
        param: str | None = None,
    ) -> str:
        json_str = json.dumps(
            self.create_error_response(
                message=message,
                err_type=err_type,
                status_code=status_code,
                param=param,
            ).model_dump()
        )
        return json_str

    def _raise_if_error(self, finish_reason: str | None, request_id: str) -> None:
        return self.create_error_response(
            str(e),
            err_type="InternalServerError",
            status_code=e.status_code,
        )

    def _convert_generation_error_to_streaming_response(
        self, e: GenerationError
    ) -> str:
        message_types = self._get_message_types(request)
        default_mm_loras = set()

        for lora in self.models.lora_requests.values():
            if lora.lora_name in message_types:
                default_mm_loras.add(lora)

        if len(default_mm_loras) == 1:
            return default_mm_loras.pop()
        return None

    def _maybe_get_adapters(
        self,
        request: AnyRequest,
        supports_default_mm_loras: bool = False,
    ) -> LoRARequest | None:
        if request.model in self.models.lora_requests:
            return self.models.lora_requests[request.model]

        if supports_default_mm_loras:
            default_mm_lora = self._get_active_default_mm_loras(request)
            if default_mm_lora is not None:
                return default_mm_lora

        if self._is_model_supported(request.model):
            return None

        # if _check_model has been called earlier, this will be unreachable
        raise ValueError(f"The model `{request.model}` does not exist.")

    def _get_message_types(self, request: AnyRequest) -> set[str]:
        message_types: set[str] = set()

        if not hasattr(request, "messages"):
            return message_types

        messages = request.messages
        if messages is None or isinstance(messages, (str, bytes)):
            return message_types

        for message in messages:
            if (
                isinstance(message, dict)
                and "content" in message
                and isinstance(message["content"], list)
            ):
                for content_dict in message["content"]:
                    if "type" in content_dict:
                        message_types.add(content_dict["type"].split("_")[0])
        return message_types

    def _validate_input(
        self,
        request: object,
        input_ids: list[int],
        input_text: str,
    ) -> TokensPrompt:
        token_num = len(input_ids)

        # Note: EmbeddingRequest, ClassificationRequest,
        # and ScoreRequest doesn't have max_tokens
        if isinstance(
            request,
            (
                EmbeddingChatRequest,
                EmbeddingCompletionRequest,
                ScoreDataRequest,
                ScoreTextRequest,
                ScoreQueriesDocumentsRequest,
                RerankRequest,
                ClassificationCompletionRequest,
                ClassificationChatRequest,
            ),
        ):
            # Note: input length can be up to the entire model context length
            # since these requests don't generate tokens.
            if token_num > self.max_model_len:
                operations: dict[type[AnyRequest], str] = {
                    ScoreDataRequest: "score",
                    ScoreTextRequest: "score",
                    ScoreQueriesDocumentsRequest: "score",
                    ClassificationCompletionRequest: "classification",
                    ClassificationChatRequest: "classification",
                }
                operation = operations.get(type(request), "embedding generation")
                raise VLLMValidationError(
                    f"This model's maximum context length is "
                    f"{self.max_model_len} tokens. However, you requested "
                    f"{token_num} tokens in the input for {operation}. "
                    f"Please reduce the length of the input.",
                    parameter="input_tokens",
                    value=token_num,
                )
            return TokensPrompt(prompt=input_text, prompt_token_ids=input_ids)

        # Note: TokenizeRequest and DetokenizeRequest doesn't have max_tokens
        # and does not require model context length validation
        if isinstance(
            request,
            (TokenizeCompletionRequest, TokenizeChatRequest, DetokenizeRequest),
        ):
            return TokensPrompt(prompt=input_text, prompt_token_ids=input_ids)

        # chat completion endpoint supports max_completion_tokens
        if isinstance(request, ChatCompletionRequest):
            # TODO(#9845): remove max_tokens when field dropped from OpenAI API
            max_tokens = request.max_completion_tokens or request.max_tokens
        else:
            max_tokens = getattr(request, "max_tokens", None)

        # Note: input length can be up to model context length - 1 for
        # completion-like requests.
        if token_num >= self.max_model_len:
            raise VLLMValidationError(
                f"This model's maximum context length is "
                f"{self.max_model_len} tokens. However, your request has "
                f"{token_num} input tokens. Please reduce the length of "
                "the input messages.",
                parameter="input_tokens",
                value=token_num,
            )

        if max_tokens is not None and token_num + max_tokens > self.max_model_len:
            raise VLLMValidationError(
                "'max_tokens' or 'max_completion_tokens' is too large: "
                f"{max_tokens}. This model's maximum context length is "
                f"{self.max_model_len} tokens and your request has "
                f"{token_num} input tokens ({max_tokens} > {self.max_model_len}"
                f" - {token_num}).",
                parameter="max_tokens",
                value=max_tokens,
            )

        return TokensPrompt(prompt=input_text, prompt_token_ids=input_ids)

    def _validate_chat_template(
        self,
        request_chat_template: str | None,
        chat_template_kwargs: dict[str, Any] | None,
        trust_request_chat_template: bool,
    ) -> ErrorResponse | None:
        if not trust_request_chat_template and (
            request_chat_template is not None
            or (
                chat_template_kwargs
                and chat_template_kwargs.get("chat_template") is not None
            )
        ):
            return self.create_error_response(
                "Chat template is passed with request, but "
                "--trust-request-chat-template is not set. "
                "Refused request with untrusted chat template."
            )
        return None

    @staticmethod
    def _prepare_extra_chat_template_kwargs(
        request_chat_template_kwargs: dict[str, Any] | None = None,
        default_chat_template_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if raw_request is not None and (
            (req_id := raw_request.headers.get("X-Request-Id")) is not None
        ):
            return req_id

        return random_uuid() if default is None else default

    @staticmethod
    def _get_data_parallel_rank(raw_request: Request | None) -> int | None:
