# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from vllm.model_executor.models import qwen3_5


def test_qwen35_runtime_flag_reads_attn_config_tuning_env(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_QWEN35_FULLATTN_STABILIZER", "0")
    inf_config = SimpleNamespace(
        tuning_env={"FASTINFERENCE_QWEN35_FULLATTN_STABILIZER": "1"}
    )

    assert qwen3_5._qwen35_config_truthy(
        inf_config,
        "FASTINFERENCE_QWEN35_FULLATTN_STABILIZER",
    )


def test_qwen35_runtime_flag_ignores_env_without_config(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_QWEN35_FULLATTN_STABILIZER", "1")

    assert not qwen3_5._qwen35_config_truthy(
        None,
        "FASTINFERENCE_QWEN35_FULLATTN_STABILIZER",
    )


def test_qwen35_sdpa_prefill_flag_reads_attn_config_tuning_env(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_QWEN35_FULLATTN_USE_SDP_PREFILL", "0")
    inf_config = SimpleNamespace(
        tuning_env={"FASTINFERENCE_QWEN35_FULLATTN_USE_SDP_PREFILL": "1"}
    )

    assert qwen3_5._qwen35_config_truthy(
        inf_config,
        "FASTINFERENCE_QWEN35_FULLATTN_USE_SDP_PREFILL",
    )


def test_qwen35_disable_ablation_flags_read_attn_config_tuning_env(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FASTINFERENCE_DISABLE_RESIDUAL_STABILIZER", "0")
    monkeypatch.setenv("FASTINFERENCE_DISABLE_LINEAR_INPUT_CAP", "0")
    inf_config = SimpleNamespace(
        tuning_env={
            "FASTINFERENCE_DISABLE_RESIDUAL_STABILIZER": "1",
            "FASTINFERENCE_DISABLE_LINEAR_INPUT_CAP": "1",
        }
    )

    assert qwen3_5._qwen35_config_truthy(
        inf_config,
        "FASTINFERENCE_DISABLE_RESIDUAL_STABILIZER",
    )
    assert qwen3_5._qwen35_config_truthy(
        inf_config,
        "FASTINFERENCE_DISABLE_LINEAR_INPUT_CAP",
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
