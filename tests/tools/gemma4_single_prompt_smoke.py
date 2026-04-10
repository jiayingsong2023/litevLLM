#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Minimal Gemma4 single-prompt smoke test (LiteEngine only).

Goal: verify one prompt can finish generation end-to-end with Gemma4 Q4 checkpoints
(e.g. compressed-tensors AWQ 4-bit), without HF alignment overhead.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

from vllm.config import CacheConfig, LoadConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.engine.lite_engine import LiteEngine
from vllm.model_executor.model_loader import get_tokenizer
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams


def _read_quant_method(model_path: str) -> str:
    cfg = Path(model_path) / "config.json"
    if not cfg.exists():
        return "unknown"
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except Exception:
        return "unknown"
    q = data.get("quantization_config") or data.get("compression_config")
    if not isinstance(q, dict):
        text = data.get("text_config")
        if isinstance(text, dict):
            q = text.get("quantization_config")
    if isinstance(q, dict):
        return str(q.get("quant_method", "unknown"))
    return "none"


def _extract_text(output: RequestOutput) -> str:
    if not output.outputs:
        return ""
    return str(getattr(output.outputs[0], "text", ""))


def _run_single_prompt(args: argparse.Namespace) -> tuple[str, int]:
    model_path = args.model

    model_cfg = ModelConfig(
        model=model_path,
        tokenizer=args.tokenizer or model_path,
    )
    cache_cfg = CacheConfig(
        block_size=args.block_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        swap_space=args.swap_space,
    )
    sched_cfg = SchedulerConfig(
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=1,
        max_model_len=args.max_model_len,
    )
    load_cfg = LoadConfig(load_format="auto")
    v_cfg = VllmConfig(model_cfg, cache_cfg, sched_cfg, load_cfg, quant_config=None)

    print("[Smoke] Loading LiteEngine...")
    t0 = time.perf_counter()
    engine = LiteEngine(v_cfg)
    t_load = time.perf_counter() - t0

    tokenizer = get_tokenizer(args.tokenizer or model_path, trust_remote_code=True)
    engine.tokenizer = tokenizer

    req_id = "gemma4_smoke"
    sp = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    prompt_tokens = tokenizer.encode(args.prompt)
    step_budget = max(64, (len(prompt_tokens) + args.max_new_tokens) * 4)

    print(
        f"[Smoke] model={model_path} quant={_read_quant_method(model_path)} "
        f"prompt_tokens={len(prompt_tokens)} max_new_tokens={args.max_new_tokens} "
        f"step_budget={step_budget} load_s={t_load:.2f}"
    )

    engine.add_request(req_id, args.prompt, sp)

    final_output: Optional[RequestOutput] = None
    step_count = 0
    while engine.active_request_count > 0 and step_count < step_budget:
        step_count += 1
        outs = engine.step()
        for out in outs:
            if out.request_id != req_id:
                continue
            if out.finished:
                final_output = out

    if final_output is None:
        raise RuntimeError(
            f"Smoke generation did not finish within step budget={step_budget}; "
            f"active_request_count={engine.active_request_count}"
        )

    text = _extract_text(final_output)
    return text, step_count


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gemma4 single prompt smoke test (LiteEngine)")
    p.add_argument("--model", type=str, required=True, help="Model path")
    p.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path (default: --model)")
    p.add_argument("--prompt", type=str, default="Hello from Gemma4.")
    p.add_argument("--max-new-tokens", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--swap-space", type=int, default=4)
    p.add_argument("--max-num-batched-tokens", type=int, default=1024)
    p.add_argument("--max-model-len", type=int, default=2048)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    try:
        text, steps = _run_single_prompt(args)
    except Exception as exc:
        print(f"[Smoke][FAIL] {type(exc).__name__}: {exc}")
        return 1

    print(f"[Smoke][PASS] steps={steps}")
    print(f"[Smoke][Output] {text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
