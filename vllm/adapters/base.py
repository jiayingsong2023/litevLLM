# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelCapabilities:
    model_type: str
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    max_model_len: int
    supports_moe: bool
    supports_fp8_kv: bool
    supports_int4_kv: bool
    supports_paged_prefill: bool
    preferred_kv_dtype: str


@dataclass(frozen=True)
class RuntimeModelPolicy:
    """Model-specific runtime preferences consumed by the lite control plane."""

    force_kv_cache_dtype: str | None = None
    force_kv_cache_dtype_when: tuple[str, ...] = ()
    force_kv_cache_dtype_reason: str | None = None
    prefill_chunk_size_high_end: int | None = None
    prefill_chunk_size_standard: int | None = None
    tuning_env_overrides: dict[str, str] = field(default_factory=dict)


class ModelAdapter(Protocol):
    model_type: str

    def detect(self, model: Any, model_config: Any) -> ModelCapabilities:
        """Infer normalized runtime capabilities from the loaded model."""

    def runtime_policy(
        self,
        model_config: Any,
        runtime_config: Any,
    ) -> RuntimeModelPolicy:
        """Return model-specific runtime policy without leaking model names into engine."""

    def install_tuning_config(self, tuning_env: dict[str, str]) -> None:
        """Install model-specific tuning config before model construction."""
