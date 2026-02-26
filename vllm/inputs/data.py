# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeAlias, cast

import torch
from typing_extensions import NotRequired, TypedDict, TypeIs, TypeVar

from vllm.sampling_params import SamplingParams

if TYPE_CHECKING:
    from vllm.multimodal.inputs import (
        MultiModalDataDict,
        MultiModalInputs,
        MultiModalUUIDDict,
    )
else:
    MultiModalDataDict = object
    MultiModalInputs = object
    MultiModalUUIDDict = object

class _CommonKeys(TypedDict):
    multi_modal_data: NotRequired[MultiModalDataDict | None]

    mm_processor_kwargs: NotRequired[dict[str, Any] | None]

    multi_modal_uuids: NotRequired[MultiModalUUIDDict]

    cache_salt: NotRequired[str]

class TextPrompt(_CommonKeys):

class TokensPrompt(_CommonKeys):

    prompt: NotRequired[str]

class EmbedsPrompt(_CommonKeys):

class DataPrompt(_CommonKeys):

    data_format: str
Set of possible schemas for a single prompt:

- A text prompt ([`str`][] or [`TextPrompt`][vllm.inputs.data.TextPrompt])
- A tokenized prompt ([`TokensPrompt`][vllm.inputs.data.TokensPrompt])
- An embeddings prompt ([`EmbedsPrompt`][vllm.inputs.data.EmbedsPrompt])

Note that "singleton" is as opposed to a data structure
which encapsulates multiple prompts, i.e. of the sort
which may be utilized for encoder/decoder models when
the user desires to express both the encoder & decoder
prompts explicitly, i.e. 
[`ExplicitEncoderDecoderPrompt`][vllm.inputs.data.ExplicitEncoderDecoderPrompt]

A prompt of type [`SingletonPrompt`][vllm.inputs.data.SingletonPrompt] may be 
employed as (1) input to a decoder-only model, (2) input to
the encoder of an encoder/decoder model, in the scenario
where the decoder-prompt is not specified explicitly, or
(3) as a member of a larger data structure encapsulating
more than one prompt, i.e. 
[`ExplicitEncoderDecoderPrompt`][vllm.inputs.data.ExplicitEncoderDecoderPrompt]
    Represents an encoder/decoder model input prompt,
    comprising an explicit encoder prompt and a decoder prompt.

    The encoder and decoder prompts, respectively, may be formatted
    according to any of the
    [`SingletonPrompt`][vllm.inputs.data.SingletonPrompt] schemas,
    and are not required to have the same schema.

    Only the encoder prompt may have multi-modal data. mm_processor_kwargs
    should be at the top-level, and should not be set in the encoder/decoder
    prompts, since they are agnostic to the encoder/decoder.

    Note that an
    [`ExplicitEncoderDecoderPrompt`][vllm.inputs.data.ExplicitEncoderDecoderPrompt]
    may not be used as an input to a decoder-only model,
    and that the `encoder_prompt` and `decoder_prompt`
    fields of this data structure themselves must be
    [`SingletonPrompt`][vllm.inputs.data.SingletonPrompt] instances.
Set of possible schemas for an LLM input, including
both decoder-only and encoder/decoder input types:

- A text prompt ([`str`][] or [`TextPrompt`][vllm.inputs.data.TextPrompt])
- A tokenized prompt ([`TokensPrompt`][vllm.inputs.data.TokensPrompt])
- An embeddings prompt ([`EmbedsPrompt`][vllm.inputs.data.EmbedsPrompt])
- A single data structure containing both an encoder and a decoder prompt
  ([`ExplicitEncoderDecoderPrompt`][vllm.inputs.data.ExplicitEncoderDecoderPrompt])

    type: Literal["token"]

    cache_salt: NotRequired[str]

def token_inputs(
    prompt_token_ids: list[int],
    cache_salt: str | None = None,
) -> TokenInputs:
    inputs = TokenInputs(type="token", prompt_token_ids=prompt_token_ids)

    if cache_salt is not None:
        inputs["cache_salt"] = cache_salt

    return inputs

class EmbedsInputs(TypedDict):

    prompt_embeds: torch.Tensor
    Optional cache salt to be used for prefix caching.
The inputs in [`LLMEngine`][vllm.engine.llm_engine.LLMEngine] before they are
passed to the model executor.
This specifies the data required for decoder-only models.
    The inputs in [`LLMEngine`][vllm.engine.llm_engine.LLMEngine] before they
    are passed to the model executor.

    This specifies the required data for encoder-decoder models.

    decoder: TokenInputs | MultiModalInputs
A processed [`SingletonPrompt`][vllm.inputs.data.SingletonPrompt] which can be
passed to [`Sequence`][collections.abc.Sequence].
The outputs from [`vllm.inputs.preprocess.InputPreprocessor`][].
    Zip encoder and decoder prompts together into a list of
    [`ExplicitEncoderDecoderPrompt`][vllm.inputs.data.ExplicitEncoderDecoderPrompt]
    instances.

    `mm_processor_kwargs` may also be provided; if a dict is passed, the same
    dictionary will be used for every encoder/decoder prompt. If an iterable is
    provided, it will be zipped with the encoder/decoder prompts.

    This is used with generate() to support multi-turn streaming sessions
    where inputs are provided via an async generator.
