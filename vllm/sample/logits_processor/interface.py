# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import torch

from vllm import SamplingParams

if TYPE_CHECKING:
    from vllm.config import VllmConfig

class MoveDirectionality(Enum):
    # One-way i1->i2 req move within batch
    UNIDIRECTIONAL = auto()
    # Two-way i1<->i2 req swap within batch
    SWAP = auto()

# Batch indices of any removed requests.
RemovedRequest = int

# (index, params, prompt_tok_ids, output_tok_ids) tuples for new
# requests added to the batch.
AddedRequest = tuple[int, SamplingParams, list[int] | None, list[int]]

# (index 1, index 2, directionality) tuples representing
# one-way moves or two-way swaps of requests in batch
MovedRequest = tuple[int, int, MoveDirectionality]

@dataclass(frozen=True)
class BatchUpdate:

        Raise ValueError for invalid ones.

        The updated tensor must be returned but may be
        modified in-place.
        argmax computation in greedy sampling.
        NOTE: may or may not have the same value for all
        instances of a given LogitsProcessor subclass,
        depending on subclass implementation.
        to each forward pass.

        Args:
            batch_update: Non-None iff there have been changes
                to the batch makeup.
