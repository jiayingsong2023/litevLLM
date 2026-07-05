# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from vllm.lora.loader import LoRALoader
from vllm.model_executor.layers.lite_linear import LiteLinear


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = LiteLinear(3, 2, bias=False, prefix="proj")


def test_loader_transposes_peft_safetensors(tmp_path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"r": 2, "lora_alpha": 4, "target_modules": ["proj"]})
    )
    save_file(
        {
            "base_model.model.proj.lora_A.weight": torch.arange(
                6, dtype=torch.float32
            ).reshape(2, 3),
            "base_model.model.proj.lora_B.weight": torch.arange(
                4, dtype=torch.float32
            ).reshape(2, 2),
        },
        adapter_dir / "adapter_model.safetensors",
    )

    weights = LoRALoader(TinyModel()).load_adapter(
        lora_name="adapter-a",
        lora_path=str(adapter_dir),
    )

    assert set(weights) == {"proj"}
    assert weights["proj"].lora_a.shape == (3, 2)
    assert weights["proj"].lora_b.shape == (2, 2)
    assert torch.equal(
        weights["proj"].lora_a,
        torch.arange(6, dtype=torch.float32).reshape(2, 3).t(),
    )


def test_loader_rejects_shape_mismatch(tmp_path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"r": 2, "lora_alpha": 4, "target_modules": ["proj"]})
    )
    save_file(
        {
            "base_model.model.proj.lora_A.weight": torch.ones(2, 4),
            "base_model.model.proj.lora_B.weight": torch.ones(2, 2),
        },
        adapter_dir / "adapter_model.safetensors",
    )

    with pytest.raises(ValueError, match="shape"):
        LoRALoader(TinyModel()).load_adapter(
            lora_name="adapter-a",
            lora_path=str(adapter_dir),
        )
