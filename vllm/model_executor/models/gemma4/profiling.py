# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import atexit
import time

import torch

from .config import (
    _GEMMA4_PROFILE_ENABLED,
    _GEMMA4_PROFILE_PRINTED,
    _GEMMA4_PROFILE_STATS,
    Gemma4LayerConfig,
    _gemma4_range_pop,
    _gemma4_range_push,
)


def _gemma4_profile_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _gemma4_profile_record(scope: str, elapsed_s: float) -> None:
    bucket = _GEMMA4_PROFILE_STATS.setdefault(scope, {"time_s": 0.0, "count": 0.0})
    bucket["time_s"] += float(elapsed_s)
    bucket["count"] += 1.0


class _Gemma4ProfileSpan:
    def __init__(self, scope: str, layer_config: Gemma4LayerConfig | None = None):
        self.scope = scope
        self._layer_config = layer_config or Gemma4LayerConfig()
        self._start = 0.0

    def __enter__(self) -> _Gemma4ProfileSpan:
        if self._layer_config.roctx_profile_enabled:
            _gemma4_range_push(self.scope)
        if self._layer_config.profile_enabled:
            _gemma4_profile_sync()
            self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._layer_config.profile_enabled:
            _gemma4_profile_sync()
            elapsed = time.perf_counter() - self._start
            bucket = self._layer_config.profile_stats.setdefault(
                self.scope, {"time_s": 0.0, "count": 0.0}
            )
            bucket["time_s"] += float(elapsed)
            bucket["count"] += 1.0
        if self._layer_config.roctx_profile_enabled:
            _gemma4_range_pop()


def _gemma4_profile_span(
    scope: str, layer_config: Gemma4LayerConfig | None = None
) -> _Gemma4ProfileSpan:
    return _Gemma4ProfileSpan(scope, layer_config)


def _dump_gemma4_profile(config: Gemma4LayerConfig | None = None) -> None:
    """Dump per-instance or global profile stats."""
    if config is None:
        global _GEMMA4_PROFILE_PRINTED
        if (
            (not _GEMMA4_PROFILE_ENABLED)
            or _GEMMA4_PROFILE_PRINTED
            or (not _GEMMA4_PROFILE_STATS)
        ):  # noqa: E501
            return
        _GEMMA4_PROFILE_PRINTED = True
        stats = _GEMMA4_PROFILE_STATS
        label = "[gemma4-profile]"
    else:
        if (
            not config.profile_enabled
            or not config.profile_stats
            or config.profile_printed
        ):  # noqa: E501
            return
        config.profile_printed = True
        stats = config.profile_stats
        label = "[gemma4-profile]"

    total_s = sum(v["time_s"] for v in stats.values())
    if total_s <= 0:
        return
    for scope, v in sorted(stats.items()):
        print(
            f"{label} {scope}: {v['time_s']:.4f}s ({int(v['count'])} calls, "
            f"{100.0 * v['time_s'] / total_s:.1f}%)"
        )


atexit.register(_dump_gemma4_profile)
