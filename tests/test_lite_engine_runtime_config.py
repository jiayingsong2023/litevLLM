# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


def _mock_vllm_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model="dummy",
            tokenizer="dummy",
            dtype="bfloat16",
            max_model_len=1024,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=1,
            max_num_batched_tokens=512,
        ),
        quant_config=None,
    )


def test_lite_engine_attaches_runtime_config_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.engine import lite_engine as lite_engine_module

    class StopAfterModelConfigCheck(Exception):
        pass

    class FakeAdapter:
        def runtime_policy(self, *_args: Any, **_kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace()

    def fake_get_model(*, vllm_config: Any) -> Any:
        assert getattr(vllm_config, "runtime_config", None) is not None
        raise StopAfterModelConfigCheck

    monkeypatch.setattr(
        lite_engine_module,
        "get_model_adapter",
        lambda *_args, **_kwargs: FakeAdapter(),
    )
    monkeypatch.setattr(
        lite_engine_module.LiteEngine,
        "_install_tuning_configs_for_model",
        lambda self, _policy: setattr(self, "_active_tuning_env", {}),
    )
    monkeypatch.setattr(lite_engine_module, "get_model", fake_get_model)

    with pytest.raises(StopAfterModelConfigCheck):
        lite_engine_module.LiteEngine(_mock_vllm_config())
