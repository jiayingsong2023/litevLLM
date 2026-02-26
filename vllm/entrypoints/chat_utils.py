# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import warnings
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Iterable
from functools import cached_property, lru_cache, partial
from itertools import accumulate
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeAlias, TypeVar, cast

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionContentPartRefusalParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
)
from openai.types.chat import (
    ChatCompletionContentPartParam as OpenAIChatCompletionContentPartParam,
)
from openai.types.chat import (
    ChatCompletionMessageParam as OpenAIChatCompletionMessageParam,
)
from openai.types.chat.chat_completion_content_part_input_audio_param import InputAudio
from openai.types.responses import ResponseInputImageParam
from openai_harmony import Message as OpenAIHarmonyMessage
from PIL import Image
from pydantic import BaseModel, ConfigDict, TypeAdapter

# pydantic needs the TypedDict from typing_extensions
from typing_extensions import Required, TypedDict

from vllm import envs
from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.models import SupportsMultiModal
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalDataDict, MultiModalUUIDDict
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFlatField,
    MultiModalSharedField,
    VisionChunk,
    VisionChunkImage,
    VisionChunkVideo,
)
from vllm.multimodal.media import MEDIA_CONNECTOR_REGISTRY, MediaConnector
from vllm.multimodal.processing import BaseMultiModalProcessor
from vllm.utils import random_uuid
from vllm.utils.collection_utils import is_list_of
from vllm.utils.import_utils import LazyLoader

if TYPE_CHECKING:
    import torch
    import transformers
else:
    transformers = LazyLoader("transformers", globals(), "transformers")
    torch = LazyLoader("torch", globals(), "torch")

logger = init_logger(__name__)

def __getattr__(name: str):
    if name == "resolve_hf_chat_template":
        from vllm.renderers.hf import resolve_chat_template

        warnings.warn(
            "`vllm.entrypoints.chat_utils.resolve_hf_chat_template` has been moved to "
            "`vllm.renderers.hf.resolve_chat_template`. "
            "The old name will be removed in v0.16.",
            DeprecationWarning,
            stacklevel=2,
        )

        return resolve_chat_template

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

class ChatTemplateResolutionError(ValueError):

MODALITY_PLACEHOLDERS_MAP = {
    "image": "<##IMAGE##>",
    "audio": "<##AUDIO##>",
    "video": "<##VIDEO##>",
}

class AudioURL(TypedDict, total=False):
    url: Required[str]

class ChatCompletionContentPartAudioParam(TypedDict, total=False):
    audio_url: Required[AudioURL]

    type: Required[Literal["audio_url"]]
    The image embeddings. It can be either:
    - A single base64 string.
    - A dictionary where each value is a base64 string.
    uuid: str | None

class ChatCompletionContentPartAudioEmbedsParam(TypedDict, total=False):
    audio_embeds: str | dict[str, str] | None
    type: Required[Literal["audio_embeds"]]
    User-provided UUID of a media. User must guarantee that it is properly
    generated and unique for different medias.
    Either a URL of the video or a data URL with base64 encoded video data.

class PILImage(BaseModel):

    image_pil: Image.Image
    model_config = ConfigDict(arbitrary_types_allowed=True)

class CustomChatCompletionContentPILImageParam(TypedDict, total=False):

    image_pil: PILImage | None
    uuid: str | None

class CustomChatCompletionContentSimpleImageParam(TypedDict, total=False):

    image_url: str | None
    uuid: str | None

class CustomChatCompletionContentSimpleAudioParam(TypedDict, total=False):

    audio_url: str | None

class CustomChatCompletionContentSimpleVideoParam(TypedDict, total=False):

    video_url: str | None
    uuid: str | None

class CustomThinkCompletionContentParam(TypedDict, total=False):

    thinking: Required[str]

    type: Required[Literal["thinking"]]

    role: Required[str]

    name: str

    tool_call_id: str | None

    reasoning: str | None

ChatCompletionMessageParam: TypeAlias = (
    OpenAIChatCompletionMessageParam
    | CustomChatCompletionMessageParam
    | OpenAIHarmonyMessage
)

# TODO: Make fields ReadOnly once mypy supports it
class ConversationMessage(TypedDict, total=False):
    role: Required[str]

    tool_call_id: str | None

    tool_calls: Iterable[ChatCompletionMessageToolCallParam] | None

    reasoning_content: str | None

# Passed in by user
ChatTemplateContentFormatOption = Literal["auto", "string", "openai"]

# After resolving "auto"
ChatTemplateContentFormat = Literal["string", "openai"]

ModalityStr = Literal[
    "image", "audio", "video", "image_embeds", "audio_embeds", "vision_chunk"
]
_T = TypeVar("_T")

# Backward compatibility for single item input
class _BatchedSingleItemField(MultiModalSharedField):
    pass

def _detect_field(
    tensors: list[torch.Tensor],
    mm_processor: BaseMultiModalProcessor,
):
    first_item = tensors[0]
    hidden_size = mm_processor.info.ctx.model_config.get_inputs_embeds_size()

    if (
        len(tensors) == 1
        and first_item.ndim == 3
        and first_item.shape[0] == 1
        and first_item.shape[-1] == hidden_size
    ):
        logger.warning(
            "Batched multi-modal embedding inputs are deprecated for Chat API. "
            "Please pass a separate content part for each multi-modal item."
        )
        return _BatchedSingleItemField(batch_size=1)

    first_shape = first_item.shape
    if all(t.shape == first_shape for t in tensors):
        return MultiModalBatchedField()

    size_per_item = [len(tensor) for tensor in tensors]
    slice_idxs = [0, *accumulate(size_per_item)]
    slices = [
        (slice(slice_idxs[i], slice_idxs[i + 1]),) for i in range(len(size_per_item))
    ]
    return MultiModalFlatField(slices=slices)

def _merge_embeds(
    data_items: list[dict[str, "torch.Tensor"]],
    mm_processor: BaseMultiModalProcessor,
):
    if not data_items:
        return {}

    first_keys = set(data_items[0].keys())
    if any(set(item.keys()) != first_keys for item in data_items[1:]):
        raise ValueError(
            "All dictionaries in the list of embeddings must have the same keys."
        )

    fields = {
        key: _detect_field([item[key] for item in data_items], mm_processor)
        for key in first_keys
    }
    data_merged = {
        key: field._reduce_data([item[key] for item in data_items], pin_memory=False)
        for key, field in fields.items()
    }

    try:
        # TODO: Support per-request mm_processor_kwargs
        parsed_configs = mm_processor._get_mm_fields_config(
            transformers.BatchFeature(data_merged),
            {},
        )
        parsed_fields = {key: parsed_configs[key].field for key in first_keys}
        keys_to_update = [
            key
            for key in first_keys
            if (
                fields[key] != parsed_fields[key]
                and not isinstance(fields[key], _BatchedSingleItemField)
            )
        ]

        for key in keys_to_update:
            data_merged[key] = parsed_fields[key]._reduce_data(
                [item[key] for item in data_items], pin_memory=False
            )
    except Exception:
        logger.exception(
            "Error when parsing merged embeddings. "
            "Falling back to auto-detected fields."
        )

    return data_merged

def _get_embeds_data(
    modality: str,
    data_items: list[Any],
    mm_processor: BaseMultiModalProcessor,
):
    if len(data_items) == 0:
        return data_items

    if all(item is None for item in data_items):
        return data_items

    if is_list_of(data_items, torch.Tensor):
        embeds_key = f"{modality}_embeds"
        dict_items = [{embeds_key: item} for item in data_items]
        return _merge_embeds(dict_items, mm_processor)[embeds_key]

    if is_list_of(data_items, dict):
        return _merge_embeds(data_items, mm_processor)

    raise NotImplementedError(type(data_items))

class BaseMultiModalItemTracker(ABC, Generic[_T]):

    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self._model_config = model_config

        self._items_by_modality = defaultdict[str, list[_T]](list)
        # Track original modality for each vision_chunk item (image or video)
        self._modality_order = defaultdict[str, list[str]](list)

    @cached_property
    def use_unified_vision_chunk_modality(self) -> bool:
        Add a multi-modal item to the current prompt and returns the
        placeholder string to use, if any.

        An optional uuid can be added which serves as a unique identifier of the
        media.
    if chat_template is None:
        return

    elif isinstance(chat_template, Path) and not chat_template.exists():
        raise FileNotFoundError("the supplied chat template path doesn't exist")

    elif isinstance(chat_template, str):
        JINJA_CHARS = "{}\n"
        if (
            not any(c in chat_template for c in JINJA_CHARS)
            and not Path(chat_template).exists()
        ):
            # Try to find the template in the built-in templates directory
            from vllm.transformers_utils.chat_templates.registry import (
                CHAT_TEMPLATES_DIR,
            )

            builtin_template_path = CHAT_TEMPLATES_DIR / chat_template
            if not builtin_template_path.exists():
                raise ValueError(
                    f"The supplied chat template string ({chat_template}) "
                    f"appears path-like, but doesn't exist! "
                    f"Tried: {chat_template} and {builtin_template_path}"
                )

    else:
        raise TypeError(f"{type(chat_template)} is not a valid chat template type")

def _load_chat_template(
    chat_template: Path | str | None,
    *,
    is_literal: bool = False,
) -> str | None:
    if chat_template is None:
        return None

    if is_literal:
        if isinstance(chat_template, Path):
            raise TypeError(
                "chat_template is expected to be read directly from its value"
            )

        return chat_template

    try:
        with open(chat_template) as f:
            return f.read()
    except OSError as e:
        if isinstance(chat_template, Path):
            raise

        JINJA_CHARS = "{}\n"
        if not any(c in chat_template for c in JINJA_CHARS):
            # Try to load from the built-in templates directory
            from vllm.transformers_utils.chat_templates.registry import (
                CHAT_TEMPLATES_DIR,
            )

            builtin_template_path = CHAT_TEMPLATES_DIR / chat_template
            try:
                with open(builtin_template_path) as f:
                    return f.read()
            except OSError:
                msg = (
                    f"The supplied chat template ({chat_template}) "
                    f"looks like a file path, but it failed to be opened. "
                    f"Tried: {chat_template} and {builtin_template_path}. "
                    f"Reason: {e}"
                )
                raise ValueError(msg) from e

        # If opening a file fails, set chat template to be args to
        # ensure we decode so our escape are interpreted correctly
        return _load_chat_template(chat_template, is_literal=True)

_cached_load_chat_template = lru_cache(_load_chat_template)

def load_chat_template(
    chat_template: Path | str | None,
    *,
    is_literal: bool = False,
) -> str | None:
    return _cached_load_chat_template(chat_template, is_literal=is_literal)

def _get_interleaved_text_prompt(
    placeholder_storage: dict[str, list], texts: list[str]
) -> str:
    for idx, elem in enumerate(texts):
        if elem in placeholder_storage:
            texts[idx] = placeholder_storage[elem].pop(0)

    return "\n".join(texts)

# TODO: Let user specify how to insert multimodal tokens into prompt
# (similar to chat template)
def _get_full_multimodal_text_prompt(
    placeholder_storage: dict[str, list],
    texts: list[str],
    interleave_strings: bool,
) -> str:
    Parses a given multi-modal content part based on its type.

    Args:
        part: A dict containing the content part, with a potential 'type' field.

    Returns:
        A tuple (part_type, content) where:
        - part_type: Type of the part (e.g., 'text', 'image_url').
        - content: Parsed content (e.g., text, image URL).

    Raises:
        ValueError: If the 'type' field is missing and no direct URL is found.
    structured dictionary pieces for texts and images will be
    wrapped in dictionaries, i.e., {"type": "text", "text", ...} and
    {"type": "image"}, respectively. Otherwise multimodal data will be
    handled by mm_parser, and texts will be returned as strings to be joined
    with multimodal placeholders.
