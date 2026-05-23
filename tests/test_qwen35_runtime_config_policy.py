# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from vllm.adapters.qwen3_5 import Qwen35Adapter
from vllm.model_executor.models import qwen3_5


def test_qwen35_tuning_config_rejects_migrated_production_policy_names() -> None:
    qwen3_5.set_qwen35_tuning_config(
        {
            "FASTINFERENCE_QWEN35_FULLATTN_STABILIZER": "0",
            "FASTINFERENCE_QWEN35_USE_FLA_CHUNK": "0",
            "FASTINFERENCE_DISABLE_LINEAR_INPUT_CAP": "1",
        },
        locked=True,
    )

    assert qwen3_5._QWEN35_TUNING == {}


def test_qwen35_adapter_exposes_production_model_policy() -> None:
    policy = Qwen35Adapter().runtime_policy(
        SimpleNamespace(hf_config=SimpleNamespace()),
        SimpleNamespace(),
    )

    assert policy.model_policy == {
        "fullattn_stabilizer": True,
        "fullattn_use_sdpa_prefill": True,
        "residual_stabilizer": True,
        "linear_input_cap": True,
        "fla_chunk_enabled": True,
    }


def test_qwen35_runtime_flag_reads_attn_config_model_policy(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_QWEN35_FULLATTN_STABILIZER", "0")
    inf_config = SimpleNamespace(model_policy={"fullattn_stabilizer": True})

    assert qwen3_5._qwen35_model_policy_truthy(
        inf_config,
        "fullattn_stabilizer",
    )


def test_qwen35_runtime_flag_ignores_env_without_config(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_QWEN35_FULLATTN_STABILIZER", "1")

    assert not qwen3_5._qwen35_model_policy_truthy(
        None,
        "fullattn_stabilizer",
    )


def test_qwen35_sdpa_prefill_flag_reads_attn_config_model_policy(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_QWEN35_FULLATTN_USE_SDP_PREFILL", "0")
    inf_config = SimpleNamespace(model_policy={"fullattn_use_sdpa_prefill": True})

    assert qwen3_5._qwen35_model_policy_truthy(
        inf_config,
        "fullattn_use_sdpa_prefill",
    )


def test_qwen35_stabilizer_policy_reads_attn_config_model_policy(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FASTINFERENCE_DISABLE_RESIDUAL_STABILIZER", "0")
    monkeypatch.setenv("FASTINFERENCE_DISABLE_LINEAR_INPUT_CAP", "0")
    inf_config = SimpleNamespace(
        model_policy={
            "residual_stabilizer": False,
            "linear_input_cap": False,
        }
    )

    assert not qwen3_5._qwen35_model_policy_truthy(
        inf_config,
        "residual_stabilizer",
        default=True,
    )
    assert not qwen3_5._qwen35_model_policy_truthy(
        inf_config,
        "linear_input_cap",
        default=True,
    )


def test_qwen35_residual_merge_ignores_env_without_explicit_policy(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FASTINFERENCE_DISABLE_RESIDUAL_STABILIZER", "1")
    x = torch.ones((1, 1), dtype=torch.float32)
    delta = torch.full((1, 1), 1000.0, dtype=torch.float32)

    merged = qwen3_5._residual_merge_fp16(x, delta, torch.float32, 1.0)

    assert torch.allclose(merged, torch.full((1, 1), 2.0))


class _IdentityLinear(nn.Module):
    output_size = 1

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones((1, 1), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_qwen35_litelinear_stable_ignores_env_without_explicit_policy(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FASTINFERENCE_DISABLE_LINEAR_INPUT_CAP", "1")
    layer = _IdentityLinear()
    x = torch.full((1, 1), 10.0, dtype=torch.float32)

    out = qwen3_5._call_litelinear_stable(layer, x, pre_cap=1.0)

    assert torch.allclose(out, torch.ones((1, 1), dtype=torch.float32))
