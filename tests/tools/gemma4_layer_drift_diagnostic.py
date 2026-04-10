#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Gemma4 long-decode layer drift diagnostic.

Purpose:
- reuse a fixed Gemma edge prompt pack (default: short_hi)
- force greedy long decode with ignore_eos=True
- capture per-layer outputs at selected decode checkpoints
- report local/full aggregate cosine drift against token 1 and previous checkpoint

This is a diagnostic tool, not a correctness gate.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

from vllm.config import CacheConfig, LoadConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.engine.lite_engine import LiteEngine
from vllm.model_executor.model_loader import get_tokenizer
from vllm.sampling_params import SamplingParams

_ROOT = Path(__file__).resolve().parents[2]


def _load_smoke_module() -> Any:
    p = _ROOT / "tests" / "tools" / "gemma4_single_prompt_smoke.py"
    spec = importlib.util.spec_from_file_location("gemma4_single_prompt_smoke", p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_prompts(path: str) -> list[tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: list[tuple[str, str]] = []
    for idx, item in enumerate(raw):
        if isinstance(item, str):
            out.append((f"prompt_{idx}", item))
        elif isinstance(item, dict) and "prompt" in item:
            out.append((str(item.get("id", f"prompt_{idx}")), str(item["prompt"])))
    return out


def _resolve_prompt(args: argparse.Namespace) -> tuple[str, str]:
    if args.prompt is not None:
        return "custom", args.prompt
    prompts = _load_prompts(args.prompts_file)
    for prompt_id, prompt in prompts:
        if prompt_id == args.prompt_id:
            return prompt_id, prompt
    raise ValueError(f"prompt_id={args.prompt_id!r} not found in {args.prompts_file}")


def _build_engine(args: argparse.Namespace) -> tuple[LiteEngine, Any]:
    model_cfg = ModelConfig(
        model=args.model,
        tokenizer=args.tokenizer or args.model,
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
    engine = LiteEngine(v_cfg)
    tokenizer = get_tokenizer(args.tokenizer or args.model, trust_remote_code=True)
    engine.tokenizer = tokenizer
    return engine, tokenizer


def _parse_checkpoints(text: str) -> list[int]:
    vals = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not vals:
        raise ValueError("At least one checkpoint is required")
    if 1 not in vals:
        vals = [1] + vals
    return vals


def _maybe_tuple_first(x: Any) -> Any:
    if isinstance(x, tuple):
        return x[0]
    return x


def _last_token_hidden(x: Any) -> Optional[torch.Tensor]:
    tensor = _maybe_tuple_first(x)
    if not isinstance(tensor, torch.Tensor):
        return None
    if tensor.ndim == 3:
        return tensor[:, -1, :].detach().float().cpu()
    if tensor.ndim == 2:
        return tensor[-1:, :].detach().float().cpu()
    return None


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.view(1, -1), b.view(1, -1), dim=-1).item())


def _summarize(
    step_captures: dict[int, dict[int, torch.Tensor]],
    token_to_step: dict[int, int],
    layers: list[Any],
    checkpoints: list[int],
) -> list[str]:
    lines: list[str] = []
    for cp in checkpoints:
        if cp == 1:
            continue
        by_kind: dict[str, list[float]] = defaultdict(list)
        prev_cp = max(x for x in checkpoints if x < cp)
        step_cp = token_to_step.get(cp)
        step_prev = token_to_step.get(prev_cp)
        step_t1 = token_to_step.get(1)
        for li, layer in enumerate(layers):
            if step_t1 is None or step_prev is None or step_cp is None:
                continue
            if (
                step_t1 not in step_captures[li]
                or step_prev not in step_captures[li]
                or step_cp not in step_captures[li]
            ):
                continue
            kind = "local" if bool(getattr(layer.self_attn, "is_sliding", False)) else "full"
            by_kind[f"{kind}_to_t1"].append(
                _cos(step_captures[li][step_cp], step_captures[li][step_t1])
            )
            by_kind[f"{kind}_to_prev"].append(
                _cos(step_captures[li][step_cp], step_captures[li][step_prev])
            )
        lines.append(f"\n[token={cp}]")
        for kind in ("local", "full"):
            vals_t1 = by_kind.get(f"{kind}_to_t1", [])
            vals_prev = by_kind.get(f"{kind}_to_prev", [])
            if not vals_t1:
                lines.append(f"  {kind}: no captures")
                continue
            lines.append(
                "  "
                + f"{kind}: cos_to_t1 mean={sum(vals_t1)/len(vals_t1):.6f} min={min(vals_t1):.6f} "
                + f"cos_to_prev mean={sum(vals_prev)/len(vals_prev):.6f} min={min(vals_prev):.6f}"
            )
    return lines


def _build_parser() -> argparse.ArgumentParser:
    smoke = _load_smoke_module()
    default_model = smoke.resolve_default_model_path()
    p = argparse.ArgumentParser(description="Gemma4 long-decode layer drift diagnostic")
    p.add_argument("--model", type=str, default=default_model, required=default_model is None)
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument(
        "--prompts-file",
        type=str,
        default=str(_ROOT / "tests" / "tools" / "fixtures" / "gemma4_edge_prompts_debug.json"),
    )
    p.add_argument("--prompt-id", type=str, default="short_hi")
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--checkpoints", type=str, default="16,24,32")
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--swap-space", type=int, default=4)
    p.add_argument("--max-num-batched-tokens", type=int, default=1024)
    p.add_argument("--max-model-len", type=int, default=512)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    checkpoints = _parse_checkpoints(args.checkpoints)
    prompt_id, prompt = _resolve_prompt(args)
    smoke = _load_smoke_module()
    engine, tokenizer = _build_engine(args)
    wrapped_prompt = smoke._apply_chat_template(tokenizer, prompt)
    layers = list(engine.model.model.layers)
    step_captures: dict[int, dict[int, torch.Tensor]] = {i: {} for i in range(len(layers))}
    token_to_step: dict[int, int] = {}
    handles = []
    current_step = {"value": 0}

    def _make_hook(layer_idx: int):
        def _hook(_m: Any, _inp: Any, out: Any) -> None:
            step = current_step["value"]
            hidden = _last_token_hidden(out)
            if hidden is None:
                return
            if step not in step_captures[layer_idx]:
                step_captures[layer_idx][step] = hidden

        return _hook

    for li, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(_make_hook(li)))

    sp = SamplingParams(
        max_tokens=args.max_new_tokens,
        min_tokens=1,
        temperature=args.temperature,
        top_p=args.top_p,
        ignore_eos=True,
    )
    req_id = f"gemma4_drift_{prompt_id}"
    engine.add_request(req_id, wrapped_prompt, sp)

    final_text = ""
    step_budget = max(96, args.max_new_tokens * 8)
    try:
        while engine.active_request_count > 0 and current_step["value"] < step_budget:
            current_step["value"] += 1
            outs = engine.step()
            for out in outs:
                if out.request_id != req_id:
                    continue
                if out.outputs:
                    final_text = str(getattr(out.outputs[0], "text", ""))
                    token_ids = getattr(out.outputs[0], "token_ids", None) or []
                    gen_tokens = len(token_ids)
                    if gen_tokens > 0 and gen_tokens not in token_to_step:
                        token_to_step[gen_tokens] = current_step["value"]
    finally:
        for h in handles:
            h.remove()

    if current_step["value"] >= step_budget:
        print(f"[Drift][FAIL] exceeded step budget={step_budget}")
        return 1

    print(
        f"[Drift] model={args.model} prompt_id={prompt_id} checkpoints={checkpoints} "
        f"steps={current_step['value']} max_new_tokens={args.max_new_tokens}"
    )
    print(f"[Drift] output_tail={final_text[-240:]!r}")
    print(f"[Drift] token_to_step={{{', '.join(f'{k}: {v}' for k, v in sorted(token_to_step.items()) if k in checkpoints)}}}")
    for line in _summarize(step_captures, token_to_step, layers, checkpoints):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
