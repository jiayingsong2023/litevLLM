# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any, Generic, Mapping, TypeVar

from fastapi import Request
from starlette.datastructures import Headers

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.inputs.data import TokensPrompt
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_tokenizer
from vllm.utils import random_uuid

logger = init_logger(__name__)

AnyRequest = TypeVar("AnyRequest")


def _sanitize_message(message: str) -> str:
    return message.strip()


async def _merge_async_iterators(*iterators: Any):
    indexed_iterators = [iterator.__aiter__() for iterator in iterators]
    pending = {
        index: indexed_iterators[index].__anext__()
        for index in range(len(indexed_iterators))
    }

    while pending:
        done_index: int | None = None
        for index, future in list(pending.items()):
            try:
                result = await future
                done_index = index
                yield index, result
                pending[index] = indexed_iterators[index].__anext__()
                break
            except StopAsyncIteration:
                done_index = index
                pending.pop(index, None)
                break

        if done_index is None:
            break


class GenerationError(Exception):
    """Raised when request processing fails before model execution."""


@dataclass
class ServeContext(Generic[AnyRequest]):
    request: AnyRequest
    raw_request: Request | None
    model_name: str
    request_id: str
    created_time: int = field(default_factory=lambda: int(time.time()))
    lora_request: Any | None = None
    engine_prompts: list[Any] | None = None
    result_generator: Any | None = None
    final_res_batch: list[Any] = field(default_factory=list)


class _SimpleRenderer:
    """Fallback renderer used in the Lite serving path."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def get_tokenizer(self) -> Any:
        return self._tokenizer

    async def tokenize_prompt_async(
        self,
        prompt: TokensPrompt,
        _params: Any = None,
    ) -> TokensPrompt:
        token_ids = prompt.get("prompt_token_ids")
        if token_ids is not None:
            decoded_prompt = self._tokenizer.decode(token_ids)
            return TokensPrompt(prompt=decoded_prompt, prompt_token_ids=token_ids)

        text_prompt = prompt.get("prompt", "")
        encoded_ids = self._tokenizer.encode(text_prompt)
        return TokensPrompt(prompt=text_prompt, prompt_token_ids=encoded_ids)


class OpenAIServing:
    request_id_prefix = "req"

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: Any | None,
        log_error_stack: bool = False,
    ) -> None:
        self.engine_client = engine_client
        self.models = models
        self.request_logger = request_logger
        self.log_error_stack = log_error_stack

        self.model_config = models.model_config
        self.max_model_len = getattr(self.model_config, "max_model_len", 8192)
        self.renderer = _SimpleRenderer(get_tokenizer(self.model_config))
        self.io_processor = None

    async def _check_model(self, request: Any) -> ErrorResponse | None:
        request_model = getattr(request, "model", None)
        if request_model is None:
            return None
        if self.models.is_supported(request_model):
            return None
        return self.create_error_response(
            f"The model `{request_model}` does not exist.",
            status_code=HTTPStatus.NOT_FOUND,
        )

    def _base_request_id(
        self,
        raw_request: Request | None,
        default: str | None = None,
    ) -> str:
        if raw_request is not None:
            req_id = raw_request.headers.get("X-Request-Id")
            if req_id:
                return req_id
        return default or random_uuid()

    def _log_inputs(
        self,
        request_id: str,
        inputs: Any,
        params: Any,
        lora_request: Any | None = None,
    ) -> None:
        if self.request_logger is None:
            return
        try:
            self.request_logger.log_inputs(request_id, inputs, params, lora_request)
        except Exception as exc:
            logger.warning("Request logging failed for %s: %s", request_id, exc)

    def _validate_chat_template(
        self,
        request_chat_template: str | None,
        chat_template_kwargs: dict[str, Any] | None,
        trust_request_chat_template: bool,
    ) -> ErrorResponse | None:
        if trust_request_chat_template:
            return None
        if request_chat_template is not None:
            return self.create_error_response(
                "Refused request chat template because trust is disabled."
            )
        if chat_template_kwargs and chat_template_kwargs.get("chat_template") is not None:
            return self.create_error_response(
                "Refused request chat template kwargs because trust is disabled."
            )
        return None

    async def _preprocess_chat(
        self,
        request: Any,
        messages: Any,
        **_kwargs: Any,
    ) -> tuple[str, list[TokensPrompt]]:
        if not isinstance(messages, list):
            raise ValueError("messages must be a list for chat requests")

        prompt_segments: list[str] = []
        for message in messages:
            if isinstance(message, dict):
                content = message.get("content", "")
            else:
                content = getattr(message, "content", "")
            prompt_segments.append(str(content))

        final_prompt = "\n".join(prompt_segments)
        prompt_ids = self.renderer.get_tokenizer().encode(final_prompt)
        validated = self._validate_input(request, prompt_ids, final_prompt)
        return final_prompt, [validated]

    async def _preprocess_completion(
        self,
        request: Any,
        prompt_input: Any,
        prompt_embeds: Any = None,
    ) -> list[TokensPrompt]:
        del prompt_embeds
        tokenizer = self.renderer.get_tokenizer()

        if isinstance(prompt_input, str):
            prompt_ids = tokenizer.encode(prompt_input)
            return [self._validate_input(request, prompt_ids, prompt_input)]

        if isinstance(prompt_input, list) and prompt_input and isinstance(prompt_input[0], int):
            prompt_ids = [int(token_id) for token_id in prompt_input]
            prompt_text = tokenizer.decode(prompt_ids)
            return [self._validate_input(request, prompt_ids, prompt_text)]

        if isinstance(prompt_input, list):
            normalized_prompts: list[TokensPrompt] = []
            for item in prompt_input:
                if isinstance(item, str):
                    prompt_ids = tokenizer.encode(item)
                    normalized_prompts.append(
                        self._validate_input(request, prompt_ids, item)
                    )
                elif isinstance(item, list):
                    token_ids = [int(token_id) for token_id in item]
                    prompt_text = tokenizer.decode(token_ids)
                    normalized_prompts.append(
                        self._validate_input(request, token_ids, prompt_text)
                    )
                else:
                    raise ValueError("Unsupported prompt item type")
            return normalized_prompts

        raise ValueError("Unsupported prompt input type")

    async def _preprocess(self, ctx: ServeContext[AnyRequest]) -> ErrorResponse | None:
        prompt_input = getattr(ctx.request, "input", None)
        if prompt_input is None:
            prompt_input = getattr(ctx.request, "prompt", None)
        if prompt_input is None:
            return self.create_error_response("Request prompt is missing")
        ctx.engine_prompts = await self._preprocess_completion(ctx.request, prompt_input)
        return None

    def _create_pooling_params(
        self,
        ctx: ServeContext[AnyRequest],
    ) -> Any | ErrorResponse:
        if hasattr(ctx.request, "to_pooling_params"):
            return ctx.request.to_pooling_params()
        return self.create_error_response("Request type does not support pooling")

    async def _prepare_generators(self, ctx: ServeContext[AnyRequest]) -> ErrorResponse | None:
        if ctx.engine_prompts is None:
            return self.create_error_response("Engine prompts not available")
        if not hasattr(self.engine_client, "encode"):
            return self.create_error_response("Current engine does not support pooling")

        pooling_params = self._create_pooling_params(ctx)
        if isinstance(pooling_params, ErrorResponse):
            return pooling_params

        trace_headers = (
            None
            if ctx.raw_request is None
            else await self._get_trace_headers(ctx.raw_request.headers)
        )

        generators = []
        for index, engine_prompt in enumerate(ctx.engine_prompts):
            request_id = f"{ctx.request_id}-{index}"
            generator = self.engine_client.encode(
                engine_prompt,
                pooling_params,
                request_id,
                lora_request=ctx.lora_request,
                trace_headers=trace_headers,
                priority=getattr(ctx.request, "priority", 0),
            )
            generators.append(generator)

        ctx.result_generator = _merge_async_iterators(*generators)
        return None

    async def _collect_batch(self, ctx: ServeContext[AnyRequest]) -> ErrorResponse | None:
        if ctx.engine_prompts is None:
            return self.create_error_response("Engine prompts not available")
        if ctx.result_generator is None:
            return self.create_error_response("Result generator not available")

        final_batch: list[Any | None] = [None] * len(ctx.engine_prompts)
        async for index, result in ctx.result_generator:
            final_batch[index] = result

        if any(result is None for result in final_batch):
            return self.create_error_response("Failed to generate results for all prompts")

        ctx.final_res_batch = [result for result in final_batch if result is not None]
        return None

    def _build_response(self, ctx: ServeContext[AnyRequest]) -> Any:
        return ctx.final_res_batch

    async def handle(self, ctx: ServeContext[AnyRequest]) -> Any:
        if error := await self._check_model(ctx.request):
            return error

        preprocess_ret = await self._preprocess(ctx)
        if isinstance(preprocess_ret, ErrorResponse):
            return preprocess_ret

        generators_ret = await self._prepare_generators(ctx)
        if isinstance(generators_ret, ErrorResponse):
            return generators_ret

        collect_ret = await self._collect_batch(ctx)
        if isinstance(collect_ret, ErrorResponse):
            return collect_ret

        return self._build_response(ctx)

    async def _get_trace_headers(self, _headers: Headers) -> Mapping[str, str] | None:
        return None

    def _validate_input(
        self,
        request: Any,
        input_ids: list[int],
        input_text: str,
    ) -> TokensPrompt:
        token_num = len(input_ids)
        max_tokens = getattr(request, "max_tokens", None)
        if max_tokens is None:
            max_tokens = getattr(request, "max_completion_tokens", None)

        if token_num >= self.max_model_len:
            raise ValueError(
                f"Input too long: {token_num}, model max length is {self.max_model_len}."
            )

        if max_tokens is not None and token_num + max_tokens > self.max_model_len:
            raise ValueError(
                f"max_tokens={max_tokens} exceeds remaining context "
                f"({self.max_model_len - token_num})."
            )

        return TokensPrompt(prompt=input_text, prompt_token_ids=input_ids)

    def _maybe_get_adapters(self, request: Any) -> Any | None:
        request_model = getattr(request, "model", None)
        if request_model is not None and request_model in self.models.lora_requests:
            return self.models.lora_requests[request_model]
        return None

    def create_error_response(
        self,
        message: str | Exception,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
        param: str | None = None,
    ) -> ErrorResponse:
        if isinstance(message, Exception):
            message = str(message)

        return ErrorResponse(
            error=ErrorInfo(
                message=_sanitize_message(message),
                type=err_type,
                code=int(status_code),
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
        return json.dumps(
            self.create_error_response(
                message=message,
                err_type=err_type,
                status_code=status_code,
                param=param,
            ).model_dump()
        )
