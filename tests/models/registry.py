# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping, Set
from dataclasses import dataclass, field
from typing import Any, Literal

import pytest
from packaging.version import Version
from transformers import __version__ as TRANSFORMERS_VERSION

from vllm.config.model import ModelDType, TokenizerMode

@dataclass(frozen=True)
class _HfExamplesInfo:
    default: str

    tokenizer: str | None = None

    speculative_model: str | None = None

    speculative_method: str | None = None

    min_transformers_version: str | None = None

    max_transformers_version: str | None = None

    transformers_version_reason: dict[Literal["vllm", "hf"], str] | None = None

    require_embed_inputs: bool = False

    dtype: ModelDType = "auto"

    enforce_eager: bool = False

    is_available_online: bool = True

    trust_remote_code: bool = False

    max_model_len: int | None = None

    max_num_batched_tokens: int | None = None

    revision: str | None = None

    max_num_seqs: int | None = None
    If True, use the original number of layers from the model config 
    instead of minimal layers for testing.
        If the installed transformers version does not meet the requirements,
        perform the given action.
        If the model is not available online, perform the given action.
