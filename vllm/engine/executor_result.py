# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import NamedTuple

import torch


class TokenPrefillResult(NamedTuple):
    next_token_ids: torch.Tensor
    prefilled_tokens: list[int]
    is_last_chunk: list[bool]


class TokenDecodeResult(NamedTuple):
    next_token_ids: torch.Tensor
