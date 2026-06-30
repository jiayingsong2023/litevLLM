# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any, Protocol

from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams


class DirectRuntime(Protocol):
    backend_name: str

    def prepare(self) -> None:
        """Prepare model-specific direct serving state."""

    def generate(
        self,
        *,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
        lora_request: Any | None = None,
        multi_modal_data: dict[str, Any] | None = None,
    ) -> RequestOutput:
        """Generate one complete output without the generic Lite scheduler."""

    def stats(self) -> dict[str, Any]:
        """Return direct runtime stats."""

    def reset_stats(self, *, clear_prefix_cache: bool = False) -> None:
        """Reset direct runtime stats."""
