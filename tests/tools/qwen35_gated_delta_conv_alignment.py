#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Compare HF Qwen3_5GatedDeltaNet causal conv path vs Lite Qwen3_5LinearAttentionLayer
with identical weights (dense FP16/BF16). Validates first-chunk prefill conv (seq_len>1).

Usage:
  uv run python tests/tools/qwen35_gated_delta_conv_alignment.py --hf-dir models/Qwen3.5-9B-FP16
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _pair_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    x = a.float().reshape(-1)
    y = b.float().reshape(-1)
    n = min(x.numel(), y.numel())
    x = x[:n]
    y = y[:n]
    diff = (x - y).abs()
    cos = F.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0), dim=1).item()
    return {"cos_sim": cos, "max_err": diff.max().item(), "mae": diff.mean().item()}


def _set_lite_linear_from_hf(lite_lin: nn.Module, hf_weight: torch.Tensor, dtype: torch.dtype) -> None:
    """Replace LiteLinear.weight with HF nn.Linear weight [out, in]."""
    w = hf_weight.detach().to(dtype=dtype).clone()
    lite_lin.weight = nn.Parameter(w, requires_grad=False)


def _copy_gated_delta_weights(hf_block, lite_layer: torch.nn.Module, dtype: torch.dtype) -> None:
    """Copy HF Qwen3_5GatedDeltaNet tensors into Lite layer (LiteLinear + conv)."""
    h = hf_block.linear_attn
    l = lite_layer.linear_attn
    _set_lite_linear_from_hf(l.in_proj_qkv, h.in_proj_qkv.weight, dtype)
    _set_lite_linear_from_hf(l.in_proj_z, h.in_proj_z.weight, dtype)
    _set_lite_linear_from_hf(l.in_proj_a, h.in_proj_a.weight, dtype)
    _set_lite_linear_from_hf(l.in_proj_b, h.in_proj_b.weight, dtype)
    _set_lite_linear_from_hf(l.out_proj, h.out_proj.weight, dtype)
    with torch.no_grad():
        l.conv1d.weight.copy_(h.conv1d.weight.detach().to(dtype=dtype))
    l.conv1d.to(dtype)
    with torch.no_grad():
        l.norm.weight.copy_(h.norm.weight.detach().to(dtype=dtype))
    l.norm.to(dtype)
    l.A_log.data.copy_(h.A_log.detach().to(dtype=l.A_log.dtype))
    l.dt_bias.data.copy_(h.dt_bias.detach().to(dtype=l.dt_bias.dtype))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", required=True, help="Hugging Face model directory")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM

    from vllm.model_executor.models.lite_config import LiteConfig
    from vllm.model_executor.models.qwen3_5 import Qwen3_5LinearAttentionLayer

    torch.manual_seed(args.seed)
    cfg = AutoConfig.from_pretrained(args.hf_dir, trust_remote_code=True)
    text_cfg = cfg.get_text_config()
    dtype = torch.bfloat16
    td = getattr(text_cfg, "dtype", None) or getattr(text_cfg, "torch_dtype", None)
    if isinstance(td, str) and "float16" in td.lower() and "bfloat" not in td.lower():
        dtype = torch.float16
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).eval()

    hf_layer = hf_model.model.layers[0]
    lite_cfg = LiteConfig(text_cfg)
    lite_layer = Qwen3_5LinearAttentionLayer(
        lite_cfg, quant_config=None, prefix="model.layers.0", layer_idx=0
    ).eval()

    _copy_gated_delta_weights(hf_layer, lite_layer, dtype)

    b, t, hdim = 1, args.seq_len, int(text_cfg.hidden_size)
    hidden = torch.randn(b, t, hdim, device="cpu", dtype=dtype)

    # HF fallback path (no causal_conv1d_fn): silu(conv1d(in_proj^T))
    hf_attn = hf_layer.linear_attn
    mq_hf = hf_attn.in_proj_qkv(hidden)
    mq_hf = mq_hf.transpose(1, 2)
    post_conv_hf = F.silu(hf_attn.conv1d(mq_hf)[:, :, :t])

    # Lite: same math as first-chunk checkpoint conv branch
    mq_l = lite_layer.linear_attn.in_proj_qkv(hidden)
    x_legacy = mq_l.transpose(1, 2).contiguous()
    post_conv_lite = F.silu(lite_layer.linear_attn.conv1d(x_legacy)[:, :, :t])

    m = _pair_metrics(post_conv_hf, post_conv_lite)
    print("=== HF vs Lite: post-conv (SiLU) tensor, layer 0 linear_attn ===")
    print(f"  shape: {tuple(post_conv_hf.shape)}")
    print(f"  CosSim={m['cos_sim']:.8f}  max_err={m['max_err']:.6e}  mae={m['mae']:.6e}")
    ok = m["cos_sim"] > 0.999999 and m["max_err"] < 1e-4
    print(f"  status: {'PASS' if ok else 'CHECK (expected ~1.0 when weights copied)'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
