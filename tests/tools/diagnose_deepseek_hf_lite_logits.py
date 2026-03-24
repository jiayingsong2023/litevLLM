#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Diagnose where HF (bf16) vs Lite (GGUF) last-token logits differ, and how to improve alignment.

Gap sources (typical order of impact):
  1) Quantization: Q4_K_M dequant != bf16 reference — dominates full-vocab cosine. Mitigation: use
     Q8_0 / Q6_K / F16 GGUF, or compare Lite vs HF with the same checkpoint in bf16 (safetensors).
  2) MoE router numerics: HF uses float32 matmul for gate; Lite matches this when using current
     DeepseekV2MoE (F.linear(x.float(), w.float())).
  3) Implementation: shared RoPE on DeepseekV2Model, MLA decode self-check — verify with
     compare_hf_lite_deepseek_logits.py --check-mla-decode.

Metrics: prefer top-1 / top-5 agreement and MSE on argmax neighborhood over raw cosine on full
vocab (cosine can be negative when both vectors are noisy and not colinear).

Example:
  uv run python tests/tools/diagnose_deepseek_hf_lite_logits.py \\
    --lite-model models/DeepSeek-V2-Lite-GGUF \\
    --hf-model models/DeepSeek-V2-Lite-Chat \\
    --chat-template auto

Per-layer hidden comparison (HF vs Lite, last token):
  uv run python tests/tools/compare_hf_lite_deepseek_layer_hiddens.py \\
    --lite-model models/DeepSeek-V2-Lite-GGUF \\
    --hf-model models/DeepSeek-V2-Lite-Chat \\
    --chat-template auto
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main() -> int:
    p = argparse.ArgumentParser(description="HF vs Lite logits diagnosis (DeepSeek V2 Lite)")
    p.add_argument("--lite-model", type=str, required=True)
    p.add_argument("--hf-model", type=str, default=None)
    p.add_argument("--prompt", type=str, default="What is 2+2? One digit.")
    p.add_argument("--chat-template", choices=("off", "auto", "on"), default="off")
    p.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    p.add_argument("--max-model-len", type=int, default=2048)
    args = p.parse_args()

    import importlib.util

    _cmp_path = os.path.join(os.path.dirname(__file__), "compare_hf_lite_deepseek_logits.py")
    _spec = importlib.util.spec_from_file_location("compare_hf_lite_deepseek_logits", _cmp_path)
    if _spec is None or _spec.loader is None:
        print("[Error] cannot load compare_hf_lite_deepseek_logits.py")
        return 2
    cmp = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(cmp)

    _apply_chat_template_if_needed = cmp._apply_chat_template_if_needed
    _default_hf_model = cmp._default_hf_model
    _encode = cmp._encode
    _resolved_tokenizer_path = cmp._resolved_tokenizer_path
    _topk_diff = cmp._topk_diff
    run_hf = cmp.run_hf
    run_lite = cmp.run_lite
    from vllm.model_executor.models.deepseek_v2 import patch_deepseek_config_json_for_tokenizer
    from transformers import AutoTokenizer

    hf_model = args.hf_model or _default_hf_model()
    tok_path = _resolved_tokenizer_path(args.lite_model, None)
    try:
        patch_deepseek_config_json_for_tokenizer(tok_path)
    except Exception:
        pass
    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    text = _apply_chat_template_if_needed(tok, args.prompt, args.chat_template)
    input_ids = _encode(tok, text)
    if input_ids.shape[1] == 0:
        print("[Error] empty input_ids")
        return 2
    cpu = input_ids.cpu()

    print("[1] Loading HF (reference)...")
    logits_hf = run_hf(hf_model, tok_path, cpu, args.dtype)
    print("[2] Loading Lite (GGUF)...")
    logits_lite = run_lite(args.lite_model, cpu, args.max_model_len)

    stats = _topk_diff(logits_hf, logits_lite, k=32)
    print("\n=== Last-token logits (HF vs Lite) ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Neighbourhood around HF argmax (less sensitive to global rotation than full cosine)
    ia = int(logits_hf.argmax().item())
    lo = max(0, ia - 8)
    hi = min(int(logits_hf.numel()), ia + 9)
    a = logits_hf[lo:hi]
    b = logits_lite[lo:hi]
    cos_local = float(
        (a.float() @ b.float()) / (a.float().norm() * b.float().norm() + 1e-8)
    )
    print(f"\n  local_cosine_[argmax±8]: {cos_local:.6f}")

    print(
        "\n--- How to improve alignment ---\n"
        "  • Strongest: use higher-bit GGUF (Q8 / Q6_K) or bf16 safetensors for parity tests.\n"
        "  • Keep MoE router in fp32 (Lite DeepseekV2MoE matches HF gate linear dtype).\n"
        "  • Track top-1 match + local cosine; full-vocab cosine is harsh under Q4.\n"
        "  • Run: uv run python tests/tools/compare_hf_lite_deepseek_logits.py "
        "--check-mla-decode (MLA path vs full prefill on Lite).\n"
        "  • Layer-wise hidden (where HF vs Lite diverges): "
        "uv run python tests/tools/compare_hf_lite_deepseek_layer_hiddens.py "
        "--lite-model <GGUF_dir> --hf-model <HF_dir> --chat-template auto\n"
        "  • Numeric knobs (may tighten logits vs HF; Q4 still won't match bf16 exactly): "
        "FASTINFERENCE_DEEPSEEK_ATTN_FP32=1 (FP32 QK^T in MLA), "
        "FASTINFERENCE_GGUF_DEQUANT_FP32=1 (FP32 GGUF dequant).\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
