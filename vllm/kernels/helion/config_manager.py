# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path
from typing import Any

from vllm.logger import init_logger
from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "ConfigManager requires helion to be installed. "
        "Install it with: pip install helion"
    )

import helion

logger = init_logger(__name__)

class ConfigSet:

    _instance: "ConfigManager | None" = None
    _instance_base_dir: Path | None = None

    def __new__(cls, base_dir: str | Path | None = None) -> "ConfigManager":
        resolved_base_dir = cls._resolve_base_dir(base_dir)

        if cls._instance is not None:
            # Instance already exists - check for base_dir mismatch
            if cls._instance_base_dir != resolved_base_dir:
                raise ValueError(
                    f"ConfigManager singleton already exists with base_dir "
                    f"'{cls._instance_base_dir}', cannot create with different "
                    f"base_dir '{resolved_base_dir}'"
                )
            return cls._instance

        # Create new instance
        instance = super().__new__(cls)
        cls._instance = instance
        cls._instance_base_dir = resolved_base_dir
        return instance

    def __init__(self, base_dir: str | Path | None = None):
        # Only initialize if not already initialized
        if hasattr(self, "_base_dir"):
            return

        self._base_dir = self._resolve_base_dir(base_dir)
        logger.debug("ConfigManager initialized with base_dir: %s", self._base_dir)

    @staticmethod
    def _resolve_base_dir(base_dir: str | Path | None) -> Path:
        if base_dir is not None:
            return Path(base_dir).resolve()
        return (Path(__file__).parent / "configs").resolve()

    @classmethod
    def get_instance(cls) -> "ConfigManager":
        if cls._instance is None:
            raise RuntimeError(
                "ConfigManager instance has not been created. "
                "Call ConfigManager(base_dir=...) first to initialize."
            )
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
