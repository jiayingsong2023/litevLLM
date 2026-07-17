# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import torch

from vllm.utils.text_utils import truthy

_GEMMA4_ALLOWED_TUNING_ENV = frozenset(
    {
        "FASTINFERENCE_GEMMA4_LAYER_PROFILE",
        "FASTINFERENCE_GEMMA4_ROCTX_PROFILE",
    }
)
_GEMMA4_TUNING: dict[str, str] = {}
_GEMMA4_TUNING_LOCKED = False


@dataclass
class Gemma4LayerConfig:
    """Per-instance tuning/profile configuration for Gemma4 layers.

    Replaces the module-level globals ``_GEMMA4_TUNING``,
    ``_GEMMA4_PROFILE_ENABLED``, ``_GEMMA4_ROCTX_PROFILE_ENABLED``,
    ``_GEMMA4_PROFILE_STATS``, ``_GEMMA4_PROFILE_PRINTED``, and
    ``_GEMMA4_ROPE_CACHE_POOL``.

    Instance lifetime is tied to the owning ``Gemma4DecoderLayer``.
    """

    profile_enabled: bool = False
    roctx_profile_enabled: bool = False
    profile_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    profile_printed: bool = False
    tuning: dict[str, str] = field(default_factory=dict)
    tuning_locked: bool = False
    rope_cache_pool: OrderedDict[
        tuple[int, int, float, str, float, str, int, str],
        tuple[torch.Tensor, torch.Tensor],
    ] = field(  # noqa: E501
        default_factory=OrderedDict
    )


def set_gemma4_tuning_config(
    values: dict[str, object] | None, *, locked: bool = False
) -> Gemma4LayerConfig:
    """Build a Gemma4LayerConfig from tuning overrides.

    Returns a new ``Gemma4LayerConfig`` instance -- no module-level side effects.
    The returned config can be installed on a ``Gemma4DecoderLayer`` via
    ``layer.set_config(config)``.

    For backward compatibility, callers that need the old module-level mutation
    can use ``_apply_global_tuning_config(config)``.
    """
    tuning = {
        str(key): str(value)
        for key, value in (values or {}).items()
        if str(key) in _GEMMA4_ALLOWED_TUNING_ENV and value is not None
    }
    profile_enabled = truthy(tuning.get("FASTINFERENCE_GEMMA4_LAYER_PROFILE"))
    roctx_enabled = truthy(tuning.get("FASTINFERENCE_GEMMA4_ROCTX_PROFILE"))

    return Gemma4LayerConfig(
        tuning=tuning,
        tuning_locked=bool(locked),
        profile_enabled=profile_enabled,
        roctx_profile_enabled=roctx_enabled,
    )


# Backward-compat shim: keep module-level mutation for callers
# that don't yet use the instance config path.
def _apply_global_tuning_config(config: Gemma4LayerConfig) -> None:
    """Apply a Gemma4LayerConfig to the legacy module-level globals.

    Deprecated: new code should pass the config to Gemma4DecoderLayer
    via ``set_config()``.
    """
    global _GEMMA4_TUNING, _GEMMA4_TUNING_LOCKED
    global _GEMMA4_PROFILE_ENABLED, _GEMMA4_ROCTX_PROFILE_ENABLED
    _GEMMA4_TUNING = dict(config.tuning)
    _GEMMA4_TUNING_LOCKED = config.tuning_locked
    _GEMMA4_PROFILE_ENABLED = config.profile_enabled
    _GEMMA4_ROCTX_PROFILE_ENABLED = config.roctx_profile_enabled


_GEMMA4_PROFILE_ENABLED = False
_GEMMA4_ROCTX_PROFILE_ENABLED = False
_GEMMA4_PROFILE_STATS: dict[str, dict[str, float]] = {}
_GEMMA4_PROFILE_PRINTED = False
_GEMMA4_MOE_MATERIALIZE_BATCH_EXPERTS = 8
_GEMMA4_ROPE_CACHE_POOL: OrderedDict[
    tuple[int, int, float, str, float, str, int, str],
    tuple[torch.Tensor, torch.Tensor],
] = OrderedDict()

try:
    from torch.cuda import nvtx as _gemma4_roctx

    _gemma4_range_push = _gemma4_roctx.range_push
    _gemma4_range_pop = _gemma4_roctx.range_pop
except Exception:  # pragma: no cover - best-effort profiling hook

    def _gemma4_range_push(name: str) -> None:
        return None

    def _gemma4_range_pop() -> None:
        return None
