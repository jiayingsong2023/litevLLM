#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Lightweight Gemma4 A-lite audit (LiteEngine only).

Goals:
- verify text-only Gemma4 Q4 checkpoints load end-to-end
- run a small fixed prompt pack under greedy decoding
- check basic key points: first token exists, generation finishes, output is non-empty
  and not obviously degenerate

This is intentionally lighter than HF parity. It is meant for >14B default audits.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from vllm.config import (
    CacheConfig,
    LoadConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.engine.lite_engine import LiteEngine
from vllm.model_executor.model_loader import get_tokenizer
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams

DEFAULT_A_LITE_PROMPTS: list[tuple[str, str]] = [
    ("en_bst", "Summarize what a binary search tree is in one short paragraph."),
    ("zh_capital", "法国的首都是哪里？请用一句话回答。"),
    (
        "chat_intro",
        "Please introduce yourself briefly and list two ways you can help "
        "with coding tasks.",
    ),
]

DEFAULT_MODEL_PATH_CANDIDATES: tuple[str, ...] = (
    "models/gemma-4-31B-it-AWQ-4bit",
    "models/Gemma-4-31B-Q4",
    "models/Gemma-4-31B-AWQ",
    "models/Gemma-4-31B-AWQ-4bit",
)


def resolve_default_model_path() -> str | None:
    env_model = os.environ.get("MODEL_GEMMA4_31B_Q4", "").strip()
    if env_model:
        return env_model
    for candidate in DEFAULT_MODEL_PATH_CANDIDATES:
        if os.path.isdir(candidate):
            return candidate
    return None


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


def _extract_token_ids(output: RequestOutput) -> list[int]:
    if not output.outputs:
        return []
    raw = getattr(output.outputs[0], "token_ids", None)
    if raw is None:
        return []
    return [int(t) for t in raw]


def _looks_like_preformatted_chat(text: str) -> bool:
    s = text.lstrip()
    if len(s) >= 12 and "<|im_start|>" in s[:400]:
        return True
    return s.startswith("<|") and "user" in s[:120].lower()


def _apply_chat_template(tokenizer: Any, text: str) -> str:
    if _looks_like_preformatted_chat(text):
        return text
    tpl = getattr(tokenizer, "chat_template", None)
    if not tpl:
        return text
    try:
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
    except Exception:
        return text


def _max_char_run(text: str) -> int:
    if not text:
        return 0
    best = 1
    run = 1
    prev = text[0]
    for ch in text[1:]:
        if ch == prev:
            run += 1
            best = max(best, run)
        else:
            run = 1
            prev = ch
    return best


def _audit_text(text: str, token_ids: list[int], min_output_chars: int) -> list[str]:
    issues: list[str] = []
    stripped = text.strip()
    if not token_ids:
        issues.append("missing_first_token")
    if not stripped:
        issues.append("empty_output")
        return issues
    if len(stripped) < min_output_chars:
        issues.append(f"too_short<{min_output_chars}")
    if "\ufffd" in stripped:
        issues.append("replacement_char")
    if _max_char_run(stripped) >= 24:
        issues.append("char_run>=24")
    if len(token_ids) >= 24:
        uniq_ratio = len(set(token_ids)) / max(1, len(token_ids))
        if uniq_ratio < 0.20:
            issues.append("low_token_diversity")
    return issues


def _run_single_request(
    *,
    engine: LiteEngine,
    tokenizer: Any,
    prompt_id: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    min_output_chars: int,
) -> tuple[str, int, int, list[str]]:
    wrapped_prompt = _apply_chat_template(tokenizer, prompt)
    prompt_tokens = tokenizer.encode(wrapped_prompt)
    step_budget = max(64, (len(prompt_tokens) + max_new_tokens) * 4)

    sp = SamplingParams(
        max_tokens=max_new_tokens,
        min_tokens=1,
        temperature=temperature,
        top_p=top_p,
    )
    req_id = f"gemma4_a_lite_{prompt_id}"
    engine.add_request(req_id, wrapped_prompt, sp)

    final_output: RequestOutput | None = None
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
            f"{prompt_id}: generation did not finish within step budget={step_budget}; "
            f"active_request_count={engine.active_request_count}"
        )

    text = _extract_text(final_output)
    token_ids = _extract_token_ids(final_output)
    issues = _audit_text(text, token_ids, min_output_chars=min_output_chars)
    first_token_id = token_ids[0] if token_ids else -1
    return text, first_token_id, step_count, issues


def _build_engine(args: argparse.Namespace) -> tuple[LiteEngine, Any, float]:
    model_path = args.model
    model_cfg = ModelConfig(
        model=model_path,
        tokenizer=args.tokenizer or model_path,
        max_model_len=args.max_model_len,
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

    print("[A-lite] Loading LiteEngine...")
    t0 = time.perf_counter()
    engine = LiteEngine(v_cfg)
    t_load = time.perf_counter() - t0
    tokenizer = get_tokenizer(args.tokenizer or model_path, trust_remote_code=True)
    engine.tokenizer = tokenizer
    return engine, tokenizer, t_load


def _resolve_prompts(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.prompt is not None:
        return [("custom", args.prompt)]
    return list(DEFAULT_A_LITE_PROMPTS)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gemma4 A-lite audit (LiteEngine)")
    p.add_argument("--model", type=str, required=True, help="Model path")
    p.add_argument(
        "--tokenizer", type=str, default=None, help="Tokenizer path (default: --model)"
    )
    p.add_argument(
        "--prompt", type=str, default=None, help="Optional single custom prompt"
    )
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--swap-space", type=int, default=4)
    p.add_argument("--max-num-batched-tokens", type=int, default=1024)
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument("--min-output-chars", type=int, default=8)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    prompts = _resolve_prompts(args)
    try:
        engine, tokenizer, load_s = _build_engine(args)
    except Exception as exc:
        print(f"[A-lite][FAIL] load: {type(exc).__name__}: {exc}")
        return 1

    print(
        f"[A-lite] model={args.model} quant={_read_quant_method(args.model)} "
        f"prompts={len(prompts)} max_new_tokens={args.max_new_tokens} "
        f"load_s={load_s:.2f}"
    )

    failures = 0
    for prompt_id, prompt in prompts:
        try:
            text, first_token_id, steps, issues = _run_single_request(
                engine=engine,
                tokenizer=tokenizer,
                prompt_id=prompt_id,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                min_output_chars=args.min_output_chars,
            )
        except Exception as exc:
            failures += 1
            print(f"\n[{prompt_id}] FAIL: {type(exc).__name__}: {exc}")
            continue

        status = "PASS" if not issues else "FAIL"
        if issues:
            failures += 1
        print(f"\n[{prompt_id}] {status} first_token_id={first_token_id} steps={steps}")
        if issues:
            print(f"  issues: {', '.join(issues)}")
        print(f"  output: {text!r}")

    passed = len(prompts) - failures
    print(f"\n[A-lite] summary: {passed}/{len(prompts)} prompts passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
