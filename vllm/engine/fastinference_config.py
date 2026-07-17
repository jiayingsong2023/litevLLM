# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vllm.engine.env_registry import FASTINFERENCE_ENV_REGISTRY, EnvScope

SUPPORTED_CONFIG_PROFILES = frozenset(
    {"auto", "balanced", "latency", "throughput", "accuracy", "benchmark"}
)
SUPPORTED_KV_TYPES = frozenset({"auto", "fp16", "fp8", "turbo_int4"})
_TOP_LEVEL_FIELDS = frozenset(
    {
        "profile",
        "kv_type",
        "debug",
        "log_level",
        "benchmark",
        "legacy_env",
        "model",
        "tuning_keyvals",
    }
)


@dataclass(frozen=True)
class BenchmarkConfig:
    profile: str = "default"


@dataclass(frozen=True)
class LegacyEnvConfig:
    enabled: bool = False


@dataclass(frozen=True)
class ModelLoadConfig:
    trust_remote_code: bool = False
    revision: str | None = None


@dataclass(frozen=True)
class FastInferenceConfig:
    profile: str = "auto"
    kv_type: str = "auto"
    debug: bool = False
    log_level: str = "info"
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    legacy_env: LegacyEnvConfig = field(default_factory=LegacyEnvConfig)
    model: ModelLoadConfig = field(default_factory=ModelLoadConfig)
    tuning_keyvals: dict[str, str] = field(default_factory=dict)


def _as_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a table")
    return value


def _validate_profile(value: object) -> str:
    profile = str(value or "auto").strip().lower()
    if profile not in SUPPORTED_CONFIG_PROFILES:
        raise ValueError(f"profile must be one of {sorted(SUPPORTED_CONFIG_PROFILES)}")
    return profile


def _validate_kv_type(value: object) -> str:
    kv_type = str(value or "auto").strip().lower()
    if kv_type not in SUPPORTED_KV_TYPES:
        raise ValueError(f"kv_type must be one of {sorted(SUPPORTED_KV_TYPES)}")
    return kv_type


def _validate_tuning_keyvals(raw: Mapping[str, Any]) -> dict[str, str]:
    values = {str(key): str(value) for key, value in raw.items()}
    unknown = sorted(
        key
        for key in values
        if key not in FASTINFERENCE_ENV_REGISTRY
        or FASTINFERENCE_ENV_REGISTRY[key].scope is EnvScope.REMOVED
    )
    if unknown:
        raise ValueError("Unknown tuning_keyvals: " + ", ".join(unknown))
    return values


def config_from_mapping(data: Mapping[str, Any]) -> FastInferenceConfig:
    unknown = sorted(set(data) - _TOP_LEVEL_FIELDS)
    if unknown:
        raise ValueError("Unknown FastInference config fields: " + ", ".join(unknown))

    benchmark_data = _as_mapping(data.get("benchmark"), "benchmark")
    legacy_data = _as_mapping(data.get("legacy_env"), "legacy_env")
    model_data = _as_mapping(data.get("model"), "model")
    tuning_raw = _as_mapping(data.get("tuning_keyvals"), "tuning_keyvals")
    trust_remote_code = bool(model_data.get("trust_remote_code", False))
    revision = str(model_data.get("revision", "")).strip() or None
    if trust_remote_code and revision is None:
        raise ValueError("model.revision is required when trust_remote_code is enabled")

    return FastInferenceConfig(
        profile=_validate_profile(data.get("profile", "auto")),
        kv_type=_validate_kv_type(data.get("kv_type", "auto")),
        debug=bool(data.get("debug", False)),
        log_level=str(data.get("log_level", "info")).strip().lower() or "info",
        benchmark=BenchmarkConfig(
            profile=str(benchmark_data.get("profile", "default")).strip().lower()
            or "default",
        ),
        legacy_env=LegacyEnvConfig(enabled=bool(legacy_data.get("enabled", False))),
        model=ModelLoadConfig(
            trust_remote_code=trust_remote_code,
            revision=revision,
        ),
        tuning_keyvals=_validate_tuning_keyvals(tuning_raw),
    )


def load_fastinference_config(path: str | Path | None) -> FastInferenceConfig:
    if path is None:
        return FastInferenceConfig()
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"FastInference config file not found: {config_path}")
    with config_path.open("rb") as f:
        payload = tomllib.load(f)
    return config_from_mapping(payload)


def resolve_fastinference_config(
    *,
    config: FastInferenceConfig | Mapping[str, Any] | None = None,
    path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> FastInferenceConfig:
    if isinstance(config, FastInferenceConfig):
        return config
    if isinstance(config, Mapping):
        return config_from_mapping(config)

    env = os.environ if environ is None else environ
    resolved_path = path
    if resolved_path is None:
        raw_env_path = str(env.get("FASTINFERENCE_CONFIG", "")).strip()
        resolved_path = raw_env_path or None
    return load_fastinference_config(resolved_path)
