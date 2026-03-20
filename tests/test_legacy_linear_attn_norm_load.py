# SPDX-License-Identifier: Apache-2.0
"""Tests for legacy flat `linear_attn.norm` checkpoint keys -> `norm.weight`."""

import os
import tempfile

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

import vllm.model_executor.model_loader as model_loader_mod
from vllm.model_executor.models.lite_config import LiteConfig
from vllm.model_executor.models.qwen3_5 import Qwen3_5LinearAttentionLayer


def _tiny_qwen35_linear_config():
    class C:
        pass

    c = C()
    for k, v in {
        "hidden_size": 128,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "vocab_size": 100,
        "rms_norm_eps": 1e-6,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 2,
    }.items():
        setattr(c, k, v)
    return LiteConfig(c)


def test_try_load_legacy_flat_linear_attn_norm_helper():
    fn = model_loader_mod._try_load_legacy_flat_linear_attn_norm_from_safetensors
    p = torch.zeros(8)
    sd = {"model.layers.2.linear_attn.norm": torch.ones(8)}
    assert fn("model.layers.2.linear_attn.norm.weight", p, list(sd.keys()), sd)
    assert (p == 1).all()


def test_safetensors_legacy_flat_norm_loads_into_norm_weight():
    cfg = _tiny_qwen35_linear_config()

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Qwen3_5LinearAttentionLayer(cfg, None, "model.layers.0")])

    class Top(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    top = Top()
    norm_w = top.model.layers[0].linear_attn.norm.weight
    head_dim = norm_w.shape[0]
    legacy = torch.arange(1, head_dim + 1, dtype=torch.float32)
    with tempfile.TemporaryDirectory() as tmp:
        save_file({"model.layers.0.linear_attn.norm": legacy.clone()}, os.path.join(tmp, "w.safetensors"))
        model_loader_mod._load_safetensors(top, tmp)
    assert torch.allclose(top.model.layers[0].linear_attn.norm.weight.data.float(), legacy)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SafetensorsAligner loads to cuda")
def test_safetensors_aligner_legacy_flat_norm():
    cfg = _tiny_qwen35_linear_config()

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Qwen3_5LinearAttentionLayer(cfg, None, "model.layers.0")])

    class Top(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    top = Top().cuda()
    head_dim = top.model.layers[0].linear_attn.norm.weight.shape[0]
    legacy = torch.arange(1, head_dim + 1, dtype=torch.float16, device="cuda")
    with tempfile.TemporaryDirectory() as tmp:
        save_file({"model.layers.0.linear_attn.norm": legacy.cpu().clone()}, os.path.join(tmp, "w.safetensors"))
        from vllm.model_executor.model_loader.safetensors import SafetensorsAligner

        SafetensorsAligner.load_weights(top, tmp)
    assert torch.allclose(
        top.model.layers[0].linear_attn.norm.weight.data.float().cpu(),
        legacy.float().cpu(),
    )
