# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CustomRuntimeComponents:
    """Model-owned runtime pieces that replace standard executors/KV manager."""

    prefill_executor: Any
    decode_executor: Any
    kv_block_manager: Any
    multimodal_processor: Any | None = None
