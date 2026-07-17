# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib

_CLASS_TO_MODULE = {
    "Gemma4Config": "vllm.transformers_utils.configs.gemma4",
    "Gemma4TextConfig": "vllm.transformers_utils.configs.gemma4",
    "Gemma4VisionConfig": "vllm.transformers_utils.configs.gemma4",
}
__all__ = list(_CLASS_TO_MODULE)


def __getattr__(name: str):
    try:
        return getattr(importlib.import_module(_CLASS_TO_MODULE[name]), name)
    except KeyError as exc:
        raise AttributeError(f"module 'configs' has no attribute {name!r}") from exc
