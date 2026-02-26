# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, TypeAlias, cast

from torch.nn import CosineSimilarity
from typing_extensions import Required, TypedDict

from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import (
    BaseMultiModalItemTracker,
    ChatCompletionContentPartImageEmbedsParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionContentPartVideoParam,
    ChatTemplateResolutionError,
    MultiModalItemTracker,
    _ContentPart,
    _parse_chat_message_content_part,
)
from vllm.inputs import TokensPrompt
from vllm.model_executor.models.interfaces import supports_score_template
from vllm.multimodal.inputs import MultiModalDataDict, MultiModalUUIDDict
from vllm.outputs import PoolingRequestOutput
from vllm.renderers.hf import safe_apply_chat_template
from vllm.tokenizers import TokenizerLike

ScoreContentPartParam: TypeAlias = (
    ChatCompletionContentPartImageParam
    | ChatCompletionContentPartImageEmbedsParam
    | ChatCompletionContentPartTextParam
    | ChatCompletionContentPartVideoParam
)

class ScoreMultiModalParam(TypedDict, total=False):

    content: Required[list[ScoreContentPartParam]]
    Perform architecture-specific manipulations on the input tokens.

    Note:
        This is an in-place operation.
    Return position of the first 1 or the length of the list
    if not found.
