# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import field
from typing import Any, Literal

import torch
from pydantic import ConfigDict, SkipValidation
from pydantic.dataclasses import dataclass

from vllm.config.utils import config
from vllm.utils.hashing import safe_hash

Device = Literal["auto", "cuda", "cpu", "tpu", "xpu"]

@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class DeviceConfig:
    This parameter is deprecated and will be
    removed in a future release.
    It will now be set automatically based
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
