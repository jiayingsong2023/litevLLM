# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SchedulerRuntimePolicy:
    max_decode_streak: int = 4
    max_prefill_deferrals: int = 2


@dataclass(frozen=True)
class BackendRuntimePolicy:
    max_prefix_cache_entries: int = 0
    gpu_greedy_sampling: bool = False
    gpu_greedy_max_tokens_only: bool = False
    gpu_greedy_bypass_cpu_policies: bool = False
    gpu_greedy_ignore_eos: bool = False
