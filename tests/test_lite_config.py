# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

from vllm.model_executor.models.lite_config import LiteConfig


def test_lite_config_uses_gemma4_unified_mm_embed_dim() -> None:
    cfg = LiteConfig(SimpleNamespace(mm_embed_dim=3840, patch_size=16))

    assert cfg.hidden_size == 3840
