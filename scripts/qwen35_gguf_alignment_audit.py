#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Audit GGUF dequant + layout vs Hugging Face safetensors for Qwen3.5 9B.

Usage (from repo root, with uv):
  uv run python scripts/qwen35_gguf_alignment_audit.py \\
    --gguf models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf \\
    --hf-dir models/Qwen3.5-9B-FP16

Optional conv check:
  uv run python scripts/qwen35_gguf_alignment_audit.py ... --conv-check
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import torch

# Repo root on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_hf_tensor(hf_dir: str, key: str) -> torch.Tensor:
    idx_path = os.path.join(hf_dir, "model.safetensors.index.json")
    if not os.path.isfile(idx_path):
        raise FileNotFoundError(f"Missing {idx_path}")
    with open(idx_path, "r") as f:
        weight_map: Dict[str, str] = json.load(f)["weight_map"]
    if key not in weight_map:
        raise KeyError(f"Key not in HF index: {key}")
    shard = os.path.join(hf_dir, weight_map[key])
    from safetensors.torch import load_file

    sd = load_file(shard, device="cpu")
    return sd[key]


def _cosine_and_maxerr(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    # float64 avoids spurious cosine > 1 on huge embedding vectors
    a = a.double().flatten()
    b = b.double().flatten()
    n = min(a.numel(), b.numel())
    if n == 0:
        return float("nan"), float("nan")
    a = a[:n]
    b = b[:n]
    denom = a.norm() * b.norm()
    cos = (a @ b / denom).item() if denom > 0 else float("nan")
    return cos, (a - b).abs().max().item()


def _hf_layer0_key(suffix: str) -> str:
    return f"model.language_model.layers.0.{suffix}"


def _resolve_hf_weight_key(hf_dir: str, candidates: List[str]) -> Tuple[str, torch.Tensor]:
    """Try multiple key patterns (language_model vs flat model.*)."""
    last_err: Exception | None = None
    for key in candidates:
        try:
            return key, _load_hf_tensor(hf_dir, key)
        except KeyError as e:
            last_err = e
    raise KeyError(f"None of {candidates} in HF index: {last_err}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen3.5 GGUF vs HF weight alignment audit")
    parser.add_argument("--gguf", required=True, help="Path to .gguf file")
    parser.add_argument("--hf-dir", required=True, help="HF model directory with safetensors + index")
    parser.add_argument(
        "--conv-check",
        action="store_true",
        help="Run depthwise conv1d on random input (GGUF vs HF weights)",
    )
    parser.add_argument(
        "--moe",
        action="store_true",
        help="Also audit Qwen3.5 MoE FFN tensors (gate / fused experts / shared / optional gate_sw)",
    )
    args = parser.parse_args()

    import gguf

    import vllm.model_executor.model_loader as ml

    reader = gguf.GGUFReader(args.gguf)
    tensor_map = {t.name: t for t in reader.tensors}

    # Flatten HF config (text_config keys) for Qwen3.5 GGUF linear-attn layout fix
    class _Cfg:
        pass

    hf_config = _Cfg()
    cfg_path = os.path.join(args.hf_dir, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r") as f:
            cfg_data = json.load(f)
        for k, v in cfg_data.items():
            setattr(hf_config, k, v)
        tc = cfg_data.get("text_config") or {}
        if isinstance(tc, dict):
            for k, v in tc.items():
                setattr(hf_config, k, v)

    # Expected PyTorch shapes [out, in] for LiteLinear (match qwen3_5 module definitions)
    checks = [
        ("token_embd.weight", "model.embed_tokens.weight", torch.Size([248320, 4096])),
        ("blk.0.attn_qkv.weight", "linear_attn.in_proj_qkv.weight", torch.Size([8192, 4096])),
        ("blk.0.attn_gate.weight", "linear_attn.in_proj_z.weight", torch.Size([4096, 4096])),
        ("blk.0.ssm_out.weight", "linear_attn.out_proj.weight", torch.Size([4096, 4096])),
        ("blk.0.ssm_alpha.weight", "linear_attn.in_proj_a.weight", torch.Size([32, 4096])),
        ("blk.0.ssm_beta.weight", "linear_attn.in_proj_b.weight", torch.Size([32, 4096])),
        ("blk.0.ffn_gate.weight", "mlp.gate_proj.weight", torch.Size([12288, 4096])),
        ("blk.0.ffn_up.weight", "mlp.up_proj.weight", torch.Size([12288, 4096])),
        ("blk.0.ffn_down.weight", "mlp.down_proj.weight", torch.Size([4096, 12288])),
    ]

    print("=== GGUF vs HF (language_model) weight alignment ===\n")
    ok = True
    for gguf_name, hf_suffix, target_shape in checks:
        if gguf_name not in tensor_map:
            print(f"[SKIP] {gguf_name} not in GGUF")
            continue
        g = tensor_map[gguf_name]
        src = ml._dequantize_gguf_tensor(g, "cpu", torch.float16, target_shape)
        if src is None:
            print(f"[FAIL] {gguf_name}: dequant returned None")
            ok = False
            continue
        if any(
            x in gguf_name
            for x in (
                "attn_qkv.weight",
                "attn_gate.weight",
                "ssm_out.weight",
                "ssm_alpha.weight",
                "ssm_beta.weight",
            )
        ):
            src = ml._qwen35_linear_attn_gguf_to_hf(src, gguf_name, hf_config)
        if gguf_name == "token_embd.weight":
            # HF Qwen3.5 checkpoints often use model.language_model.embed_tokens
            cand = [
                "model.language_model.embed_tokens.weight",
                "model.embed_tokens.weight",
            ]
        else:
            lm = _hf_layer0_key(hf_suffix)
            flat = f"model.layers.0.{hf_suffix}"
            cand = [lm, flat]
        try:
            hf_key, hf_w = _resolve_hf_weight_key(args.hf_dir, cand)
        except KeyError as e:
            print(f"[WARN] {gguf_name}: {e}")
            continue

        hf_w = hf_w.to(torch.float16)
        if src.shape != hf_w.shape:
            print(
                f"[SHAPE] {gguf_name}: dequant {tuple(src.shape)} vs HF {tuple(hf_w.shape)}"
            )
            ok = False
        cos, mx = _cosine_and_maxerr(src, hf_w)
        status = "OK" if cos > 0.999 and mx < 0.5 else "CHECK"
        print(f"  {gguf_name} -> {hf_key}: CosSim={cos:.6f} MaxErr={mx:.6f} [{status}]")
        if cos < 0.99:
            ok = False

    if args.conv_check:
        print("\n=== Depthwise conv1d spot-check (layer 0) ===\n")
        conv_gk = "blk.0.ssm_conv1d.weight"
        conv_hf = _hf_layer0_key("linear_attn.conv1d.weight")
        if conv_gk in tensor_map and conv_hf:
            try:
                hf_conv = _load_hf_tensor(args.hf_dir, conv_hf).to(torch.float16)
            except KeyError:
                hf_conv = _load_hf_tensor(
                    args.hf_dir, conv_hf.replace("model.language_model.", "model.")
                ).to(torch.float16)
            g = tensor_map[conv_gk]
            lite_conv = ml._dequantize_gguf_tensor(
                g, "cpu", torch.float16, torch.Size(hf_conv.shape)
            )
            if lite_conv is None:
                print("  [FAIL] ssm_conv1d dequant returned None")
            else:
                lite_conv = ml._qwen35_conv1d_channels_gguf_to_hf(lite_conv, hf_config)
                cos, mx = _cosine_and_maxerr(lite_conv, hf_conv)
                print(f"  conv1d: CosSim={cos:.6f} MaxErr={mx:.6f}")
                b, c, l = 1, 8192, 16
                x = torch.randn(b, c, l, device="cpu", dtype=torch.float16)
                y_hf = torch.nn.functional.conv1d(
                    x, hf_conv, bias=None, stride=1, padding=3, groups=c
                )
                y_lite = torch.nn.functional.conv1d(
                    x, lite_conv, bias=None, stride=1, padding=3, groups=c
                )
                c2, m2 = _cosine_and_maxerr(y_hf, y_lite)
                print(f"  conv1d output (random x): CosSim={c2:.6f} MaxErr={m2:.6f}")

    if args.moe:
        print("\n=== Qwen3.5 MoE FFN (HF-aligned names) ===\n")
        E = int(getattr(hf_config, "num_experts", 0) or 0)
        H = int(getattr(hf_config, "hidden_size", 0) or 0)
        inter = int(getattr(hf_config, "moe_intermediate_size", 0) or 0)
        sh = int(getattr(hf_config, "shared_expert_intermediate_size", 0) or 0)
        if not (E and H and inter and sh):
            print("  [SKIP] Missing num_experts/hidden_size/moe_intermediate_size/shared_expert_intermediate_size in config")
        else:

            def _cmp(gguf_name: str, hf_rel: str, tgt_shape: torch.Size, fused: torch.Tensor | None = None) -> None:
                nonlocal ok
                if fused is None and gguf_name not in tensor_map:
                    print(f"  [SKIP] {gguf_name} not in GGUF")
                    return
                src = (
                    fused
                    if fused is not None
                    else ml._dequantize_gguf_tensor(tensor_map[gguf_name], "cpu", torch.float16, tgt_shape)
                )
                if src is None:
                    print(f"  [FAIL] {gguf_name}: dequant returned None")
                    ok = False
                    return
                cand = [_hf_layer0_key(hf_rel), f"model.layers.0.{hf_rel}"]
                try:
                    hf_key, hf_w = _resolve_hf_weight_key(args.hf_dir, cand)
                except KeyError as e:
                    print(f"  [WARN] {gguf_name}: {e}")
                    return
                hf_w = hf_w.to(torch.float16)
                if src.shape != hf_w.shape:
                    print(f"  [SHAPE] {gguf_name}: dequant {tuple(src.shape)} vs HF {tuple(hf_w.shape)}")
                    ok = False
                    return
                cos, mx = _cosine_and_maxerr(src, hf_w)
                status = "OK" if cos > 0.999 and mx < 0.5 else "CHECK"
                print(f"  {gguf_name} -> {hf_key}: CosSim={cos:.6f} MaxErr={mx:.6f} [{status}]")
                if cos < 0.99:
                    ok = False

            _cmp("blk.0.ffn_gate_inp.weight", "mlp.gate.weight", torch.Size([E, H]))
            ge = "blk.0.ffn_gate_exps.weight"
            ue = "blk.0.ffn_up_exps.weight"
            if ge in tensor_map and ue in tensor_map:
                g_half = ml._dequantize_gguf_tensor(
                    tensor_map[ge], "cpu", torch.float16, torch.Size([E, inter, H])
                )
                u_half = ml._dequantize_gguf_tensor(
                    tensor_map[ue], "cpu", torch.float16, torch.Size([E, inter, H])
                )
                if g_half is not None and u_half is not None:
                    fused_gu = torch.cat([g_half, u_half], dim=1)
                    _cmp(ge, "mlp.experts.gate_up_proj", torch.Size([E, 2 * inter, H]), fused=fused_gu)
                else:
                    print("  [FAIL] ffn_gate_exps / ffn_up_exps dequant")
                    ok = False
            else:
                print("  [SKIP] MoE gate/up expert tensors missing")

            _cmp("blk.0.ffn_down_exps.weight", "mlp.experts.down_proj", torch.Size([E, H, inter]))
            _cmp("blk.0.ffn_gate_shexp.weight", "mlp.shared_expert.gate_proj.weight", torch.Size([sh, H]))
            _cmp("blk.0.ffn_up_shexp.weight", "mlp.shared_expert.up_proj.weight", torch.Size([sh, H]))
            _cmp("blk.0.ffn_down_shexp.weight", "mlp.shared_expert.down_proj.weight", torch.Size([H, sh]))

            sw_name = None
            for cand in ("blk.0.ffn_gate_sw_shexp.weight", "blk.0.ffn_gate_inp_shexp.weight"):
                if cand in tensor_map:
                    sw_name = cand
                    break
            if sw_name is not None:
                src = ml._dequantize_gguf_tensor(
                    tensor_map[sw_name], "cpu", torch.float16, torch.Size([1, H])
                )
                if src is not None:
                    try:
                        hf_key, hf_w = _resolve_hf_weight_key(
                            args.hf_dir,
                            [
                                _hf_layer0_key("mlp.shared_expert_gate.weight"),
                                "model.layers.0.mlp.shared_expert_gate.weight",
                            ],
                        )
                        hf_w = hf_w.to(torch.float16)
                        cos, mx = _cosine_and_maxerr(src, hf_w)
                        print(f"  {sw_name} -> {hf_key}: CosSim={cos:.6f} MaxErr={mx:.6f}")
                    except KeyError as e:
                        print(f"  [WARN] {sw_name}: HF key {e}")
                else:
                    print(f"  [FAIL] {sw_name}: dequant None")
                    ok = False
            else:
                print(
                    "  [INFO] blk.0.ffn_gate_sw_shexp / ffn_gate_inp_shexp not in GGUF "
                    "(optional; HF may still define shared_expert_gate)"
                )

    print("\n=== Done ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
