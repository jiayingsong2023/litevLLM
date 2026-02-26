# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from typing import Any, Literal, TypeAlias

from pydantic import ConfigDict, Field, field_validator, model_validator
from pydantic.dataclasses import dataclass

from vllm.config.utils import config
from vllm.utils.hashing import safe_hash
from vllm.attention.backends.registry import AttentionBackendEnum

@dataclass
class BaseDummyOptions:

    num_frames: int | None = Field(None, gt=0)
    width: int | None = Field(None, gt=0)
    height: int | None = Field(None, gt=0)

@dataclass(config=ConfigDict(extra="forbid"))
class ImageDummyOptions(BaseDummyOptions):

    length: int | None = Field(None, gt=0)

MMEncoderTPMode = Literal["weights", "data"]
MMCacheType = Literal["shm", "lru"]
DummyOptions: TypeAlias = (
    BaseDummyOptions | VideoDummyOptions | ImageDummyOptions | AudioDummyOptions
)

@config
@dataclass
class MultiModalConfig:
        prompt for each modality.
    Defaults to 999 for each modality.

    Legacy format (count only):
        {"image": 16, "video": 2}

    Configurable format (with options):
        {"video": {"count": 1, "num_frames": 32, "width": 512, "height": 512}, 
        "image": {"count": 5, "width": 512, "height": 512}}

    Mixed format (combining both):
        {"image": 16, "video": {"count": 1, "num_frames": 32, "width": 512, 
        "height": 512}}
    for `LLM` class, this refers to tensor inputs under `multi_modal_data`;
    for the OpenAI-compatible server, this refers to chat messages with content
    `"type": "*_embeds"`.

    WARNING: The vLLM engine may crash if incorrect shape of embeddings is passed.
    For example, to set num_frames for video, set
    e.g., image processor. Overrides for the multi-modal processor obtained
    from `transformers.AutoProcessor.from_pretrained`.

    The available overrides depend on the model that is being run.

    For example, for Phi-3-Vision:
    avoid re-processing past multi-modal inputs.

    This cache is duplicated for each API process and engine core process,
    resulting in a total memory usage of
    `mm_processor_cache_gb * (api_server_count + data_parallel_size)`.

    shared memory cache. Only effective when `mm_processor_cache_type` is
    When enabled, skips the language component of the model.

    This is usually only valid in disaggregated Encoder process.
    parallelism (TP).

    - `"weights"`: Within the same vLLM engine, split the weights of
        each layer across TP ranks. (default TP behavior)\n
    - `"data"`: Within the same vLLM engine, split the batched input data
        across TP ranks to process the data in parallel, while hosting
        the full weights on each TP rank.
        This batch-level DP is not to be confused with API request-level
        DP (which is controlled by `--data-parallel-size`).
        This is only supported on a per-model basis and falls back to
    using vision transformers. Accepts any value from
    language backbone model during engine initialization.

    This reduces engine startup time but shifts the responsibility to users for
    estimating the peak memory usage of the activation of multimodal encoder and
    Value sits in range [0;1) and determines fraction of media tokens
    from each video to be pruned.
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        Get the maximum number of input items allowed per prompt
        for the given modality (backward compatible).
        Get the configurable dummy data options for a modality.
        Returns None if no options are configured for this modality.
        Get the keyword arguments to pass to the multi-modal processor
        according to the extra arguments passed during inference.
