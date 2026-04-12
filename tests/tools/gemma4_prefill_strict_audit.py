#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Gemma4 31B A-strict manual audit.

This is a constrained, prefill-only HF parity audit intended for large models on
resource-limited single-GPU hosts:

- Lite path on GPU
- HF reference on CPU by default
- short, fixed prompt pack
- compares last prefill logits / greedy first token only

It is intentionally not part of the default >14B correctness path.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[2]


def _load_smoke_helpers():
    import importlib.util

    p = _ROOT / "tests" / "tools" / "gemma4_single_prompt_smoke.py"
    spec = importlib.util.spec_from_file_location("gemma4_single_prompt_smoke", p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DEFAULT_PROMPTS: dict[str, str] = {
    "en_capital": "The capital of France is",
    "zh_capital": "法国的首都是",
    "en_sort": "A binary search tree is",
}


def _default_model_path() -> Optional[str]:
    mod = _load_smoke_helpers()
    return mod.resolve_default_model_path()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gemma4 prefill-only strict audit")
    default_model = _default_model_path()
    p.add_argument("--model", type=str, default=default_model, required=default_model is None)
    p.add_argument("--hf-model", type=str, default=None)
    p.add_argument(
        "--preset",
        type=str,
        default="gemma4_31b_q4",
        choices=("gemma4_31b_q4", "gemma4_26b_a4b"),
    )
    p.add_argument("--hf-device", type=str, default="cuda", choices=("cpu", "cuda"))
    p.add_argument(
        "--prompt-id",
        type=str,
        default="en_capital",
        choices=tuple(DEFAULT_PROMPTS.keys()),
        help="Short fixed prompt for prefill parity.",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Override prompt text directly. If set, --prompt-id is ignored.",
    )
    p.add_argument("--max-model-len", type=int, default=256)
    p.add_argument("--max-num-batched-tokens", type=int, default=512)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    p.add_argument("--kv-type", type=str, default="turbo_int4")
    p.add_argument(
        "--drift-cos-threshold",
        type=float,
        default=0.995,
        help="Cosine threshold for first-drift-layer reporting.",
    )
    p.add_argument(
        "--quant",
        type=str,
        default="awq",
        choices=("none", "awq", "gguf"),
    )
    p.add_argument(
        "--print-cmd",
        action="store_true",
        help="Print delegated verify_semantic_integrity.py command before execution.",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    model_path = str(args.model)
    if not os.path.isdir(model_path):
        print(f"[A-strict][ERROR] local model dir required, got: {model_path}")
        return 2

    hf_model = args.hf_model or model_path
    if not os.path.isdir(hf_model):
        print(f"[A-strict][ERROR] local HF reference dir required, got: {hf_model}")
        return 2

    prompt = args.prompt or DEFAULT_PROMPTS[args.prompt_id]

    cmd = [
        sys.executable,
        "tests/verify_semantic_integrity.py",
        "--model",
        model_path,
        "--preset",
        args.preset,
        "--quant",
        args.quant,
        "--prompt",
        prompt,
        "--hf-model",
        hf_model,
        "--hf-device",
        args.hf_device,
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--prefill-only",
        "--apply-chat-template",
        "off",
        "--report-first-drift-layer",
        "--drift-cos-threshold",
        str(args.drift_cos_threshold),
    ]

    env = os.environ.copy()
    env["FASTINFERENCE_KV_TYPE"] = args.kv_type
    env["FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS"] = "1"
    env["FASTINFERENCE_KV_MAX_MODEL_LEN"] = str(args.max_model_len)

    if args.print_cmd:
        print("[A-strict] exec:", " ".join(cmd))
        print(
            "[A-strict] env:",
            f"FASTINFERENCE_KV_TYPE={env['FASTINFERENCE_KV_TYPE']}",
            f"FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS={env['FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS']}",
            f"FASTINFERENCE_KV_MAX_MODEL_LEN={env['FASTINFERENCE_KV_MAX_MODEL_LEN']}",
        )

    print(
        f"[A-strict] model={model_path} hf_model={hf_model} hf_device={args.hf_device} "
        f"prompt_id={'custom' if args.prompt else args.prompt_id} "
        f"max_model_len={args.max_model_len} max_num_batched_tokens={args.max_num_batched_tokens}"
    )

    proc = subprocess.run(
        cmd,
        cwd=str(_ROOT),
        env=env,
        check=False,
    )
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
