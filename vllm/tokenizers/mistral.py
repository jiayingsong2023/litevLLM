# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, overload

from mistral_common.protocol.instruct.request import (
    ChatCompletionRequest as MistralChatCompletionRequest,
)
from mistral_common.protocol.instruct.tool_calls import Function, Tool
from mistral_common.protocol.instruct.validator import ValidationMode
from mistral_common.tokens.tokenizers.base import (
    SpecialTokenPolicy,
    SpecialTokens,
)
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV13
from mistral_common.tokens.tokenizers.sentencepiece import (
    SentencePieceTokenizer,
)
from mistral_common.tokens.tokenizers.tekken import Tekkenizer

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.logger import init_logger

from .protocol import TokenizerLike

if TYPE_CHECKING:
    from transformers import BatchEncoding

    try:
        # Transformers v5
        from transformers.tokenization_mistral_common import MistralCommonBackend
    except ImportError:
        # Transformers v4
        from transformers.tokenization_mistral_common import (
            MistralCommonTokenizer as MistralCommonBackend,
        )

logger = init_logger(__name__)

def maybe_serialize_tool_calls(request: "MistralChatCompletionRequest"):
    # SEE: https://github.com/vllm-project/vllm/pull/9951
    # Credits go to: @gcalmettes
    # NOTE: There is currently a bug in pydantic where attributes
    # declared as iterables are replaced in in the instances by
    # pydantic-core ValidatorIterator instance. In particular, this
    # affects tool_calls defined in ChatCompletionAssistantMessageParam
    # model:
    # see:
    #   - https://github.com/pydantic/pydantic/issues/9467
    # As a result, tool_calls from assistant messages are never
    # deserialized in the request object if the tool_calls iterator is
    # not consumed. This affect messages passed to the MistralTokenizer
    # since no chat template is applied and therefore the tools_calls
    # iterator is not directly consumed.
    # Issue is tracked on Pydantic side, with resolution planned for
    # v2.11 release. In the meantime, the official workaround is to
    # consume the iterator so the tool_calls are correctly deserialized
    # in the OpenAI ChatCompletionAssistantMessageParam object
    # https://github.com/pydantic/pydantic/issues/9467#issuecomment-2442097291 # noqa: E501
    # Official Pydantic Issues:
    #   - https://github.com/pydantic/pydantic/issues/9541
    # TODO: remove when pydantic v2.11 is released
    for i, message in enumerate(request.messages):
        if message.get("role") == "assistant":
            tool_calls_validator = message.get("tool_calls", ().__iter__())
            validated_tool_calls = []
            while True:
                try:
                    tool_call = next(tool_calls_validator)  # type: ignore
                    validated_tool_calls.append(tool_call)
                except StopIteration:
                    break

            request.messages[i]["tool_calls"] = validated_tool_calls

def truncate_tool_call_ids(request: "MistralChatCompletionRequest"):
