# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeepSeekV4FlashGPUCapabilities:
    q8_linear: bool = True
    attention: bool = False
    compressed_attention: bool = False
    cache_update: bool = False
    moe: bool = False
    output: bool = False

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, enabled in (
                ("q8_linear", self.q8_linear),
                ("attention", self.attention),
                ("compressed_attention", self.compressed_attention),
                ("cache_update", self.cache_update),
                ("moe", self.moe),
                ("output", self.output),
            )
            if not enabled
        )

    @property
    def is_ready(self) -> bool:
        return not self.missing


class DeepSeekV4FlashGPUBackend:
    def __init__(
        self,
        *,
        capabilities: DeepSeekV4FlashGPUCapabilities | None = None,
    ) -> None:
        self.capabilities = capabilities or DeepSeekV4FlashGPUCapabilities()

    @property
    def is_ready(self) -> bool:
        return self.capabilities.is_ready

    @property
    def missing_kernels(self) -> tuple[str, ...]:
        return self.capabilities.missing

    def require_ready(self) -> None:
        if not self.is_ready:
            missing = ", ".join(self.missing_kernels)
            raise RuntimeError(f"DeepSeek V4 Flash missing GPU kernels: {missing}")
