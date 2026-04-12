#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Gemma4-26B local-decode window A/B drift audit report.

Outputs JSON with:
- baseline first_drift_token / first_drift_layer / first_drift_cos
- guarded  first_drift_token / first_drift_layer / first_drift_cos
- guard range used (start/span)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

from vllm.config import CacheConfig, LoadConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.engine.lite_engine import LiteEngine
from vllm.model_executor.model_loader import get_tokenizer


def _resolve_default_model_path() -> Optional[str]:
    env = os.environ.get("MODEL_GEMMA4_26B_A4B_Q4", "").strip()
    if env:
        return env
    candidates = [
        "models/gemma-4-26B-A4B-it-AWQ-4bit",
        "models/Gemma-4-26B-A4B-it-AWQ-4bit",
    ]
    for p in candidates:
        if Path(p).is_dir():
            return p
    return None


def _set_or_unset_env(name: str, value: Optional[str]) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def _build_engine(
    model_path: str,
    *,
    max_model_len: int,
    max_num_batched_tokens: int,
    gpu_memory_utilization: float,
) -> LiteEngine:
    model_cfg = ModelConfig(model=model_path, tokenizer=model_path)
    cache_cfg = CacheConfig(
        block_size=16,
        gpu_memory_utilization=gpu_memory_utilization,
        swap_space=4,
    )
    sched_cfg = SchedulerConfig(
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=1,
        max_model_len=max_model_len,
    )
    load_cfg = LoadConfig(load_format="auto")
    v_cfg = VllmConfig(model_cfg, cache_cfg, sched_cfg, load_cfg, quant_config=None)
    engine = LiteEngine(v_cfg)
    engine.tokenizer = get_tokenizer(model_path, trust_remote_code=True)
    return engine


def _last_token_hook(capture: dict[int, torch.Tensor], layer_idx: int):
    def _hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        t = output if isinstance(output, torch.Tensor) else output[0]
        capture[layer_idx] = t[:, -1, :].detach().float().cpu()

    return _hook


def _capture_last_hidden_for_layers(
    layers: list[torch.nn.Module],
    run_forward: Callable[[], Any],
) -> dict[int, torch.Tensor]:
    capture: dict[int, torch.Tensor] = {}
    hooks = [
        layer.register_forward_hook(_last_token_hook(capture, li))
        for li, layer in enumerate(layers)
    ]
    try:
        run_forward()
    finally:
        for h in hooks:
            h.remove()
    return capture


def _audit_local_decode_window(
    engine: LiteEngine,
    prompt_ids: list[int],
    *,
    token_start: int,
    token_end: int,
    cos_threshold: float,
) -> dict[str, int | float | None]:
    layers = list(engine.model.model.layers)
    block_tables = torch.arange(
        engine.num_blocks_per_seq, device="cuda", dtype=torch.int32
    ).view(1, -1)

    def _cached_forward(
        input_ids: torch.Tensor, pos_start: int, seq_len_after: int, is_prefill: bool
    ) -> torch.Tensor:
        seqlen = input_ids.shape[1]
        positions = torch.arange(
            pos_start, pos_start + seqlen, device="cuda", dtype=torch.long
        ).view(1, seqlen)
        slot_mapping = torch.arange(
            pos_start, pos_start + seqlen, device="cuda", dtype=torch.long
        )
        meta = {
            "slot_mapping": slot_mapping,
            "block_tables": block_tables,
            "seq_lens": torch.tensor([seq_len_after], device="cuda", dtype=torch.int32),
            "is_prefill": bool(is_prefill),
            "config": engine.inf_config,
            "kv_scale_cache": engine.kv_scale_caches,
        }
        return engine.model(input_ids, positions, engine.kv_caches, meta, None)

    for k_cache, v_cache in engine.kv_caches:
        k_cache.zero_()
        v_cache.zero_()
    for ks, vs in engine.kv_scale_caches:
        if ks is not None:
            ks.zero_()
        if vs is not None:
            vs.zero_()

    generated: list[int] = []
    prefill_input = torch.tensor([prompt_ids], device="cuda", dtype=torch.long)
    logits = _cached_forward(prefill_input, 0, len(prompt_ids), True)

    first_drift_token: int | None = None
    first_drift_layer: int | None = None
    first_drift_cos: float | None = None

    for step in range(1, token_end + 1):
        token = int(torch.argmax(logits[0, -1]).item())
        generated.append(token)
        token_input = torch.tensor([[token]], device="cuda", dtype=torch.long)
        pos_start = len(prompt_ids) + step - 1

        cached_cap = _capture_last_hidden_for_layers(
            layers,
            lambda: _cached_forward(token_input, pos_start, len(prompt_ids) + step, False),
        )
        full_ids = prompt_ids + generated
        full_input = torch.tensor([full_ids], device="cuda", dtype=torch.long)
        full_positions = torch.arange(
            len(full_ids), device="cuda", dtype=torch.long
        ).view(1, -1)
        ref_cap = _capture_last_hidden_for_layers(
            layers,
            lambda: engine.model(
                full_input,
                full_positions,
                [(None, None)] * len(engine.kv_caches),
                {},
                None,
            ),
        )

        if step >= token_start and first_drift_token is None:
            for li, layer in enumerate(layers):
                if not bool(getattr(layer.self_attn, "is_sliding", False)):
                    continue
                cos = float(
                    F.cosine_similarity(
                        cached_cap[li].view(1, -1),
                        ref_cap[li].view(1, -1),
                        dim=-1,
                    ).item()
                )
                if cos < cos_threshold:
                    first_drift_token = step
                    first_drift_layer = li
                    first_drift_cos = cos
                    break

        logits = _cached_forward(token_input, pos_start, len(prompt_ids) + step, False)

    return {
        "first_drift_token": first_drift_token,
        "first_drift_layer": first_drift_layer,
        "first_drift_cos": first_drift_cos,
        "token_start": token_start,
        "token_end": token_end,
        "cos_threshold": cos_threshold,
    }


def _run_leg(
    *,
    model: str,
    prompt: str,
    token_start: int,
    token_end: int,
    cos_threshold: float,
    kv_type: str,
    max_model_len: int,
    max_num_batched_tokens: int,
    gpu_memory_utilization: float,
    guard_enabled: bool,
    guard_start: Optional[int],
    guard_span: Optional[int],
) -> dict[str, Any]:
    _set_or_unset_env("FASTINFERENCE_KV_TYPE", kv_type)
    _set_or_unset_env("FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS", "1")
    _set_or_unset_env("FASTINFERENCE_KV_MAX_MODEL_LEN", str(max_model_len))
    _set_or_unset_env(
        "FASTINFERENCE_GEMMA4_26B_FP32_RESIDUAL_GUARD",
        "1" if guard_enabled else None,
    )
    _set_or_unset_env(
        "FASTINFERENCE_GEMMA4_26B_FP32_RESIDUAL_GUARD_START",
        str(guard_start) if guard_start is not None else None,
    )
    _set_or_unset_env(
        "FASTINFERENCE_GEMMA4_26B_FP32_RESIDUAL_GUARD_SPAN",
        str(guard_span) if guard_span is not None else None,
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        engine = _build_engine(
            model,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        tokenizer = engine.tokenizer
        assert tokenizer is not None
        prompt_ids = tokenizer.encode(prompt)
        if isinstance(prompt_ids, torch.Tensor):
            prompt_ids = prompt_ids.tolist()
        report = _audit_local_decode_window(
            engine,
            list(prompt_ids),
            token_start=token_start,
            token_end=token_end,
            cos_threshold=cos_threshold,
        )
        return {
            "ok": True,
            "error": None,
            "report": report,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "report": None,
        }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gemma4 decode-window A/B drift JSON report")
    default_model = _resolve_default_model_path()
    p.add_argument("--model", type=str, default=default_model, required=default_model is None)
    p.add_argument("--prompt", type=str, default="Hi,")
    p.add_argument("--token-start", type=int, default=2)
    p.add_argument("--token-end", type=int, default=16)
    p.add_argument("--cos-threshold", type=float, default=0.99)
    p.add_argument("--kv-type", type=str, default="fp16")
    p.add_argument("--max-model-len", type=int, default=256)
    p.add_argument("--max-num-batched-tokens", type=int, default=256)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    p.add_argument("--guard-span", type=int, default=3)
    p.add_argument("--json-out", type=str, default="")
    p.add_argument("--pretty", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    if not Path(args.model).exists():
        print(f"[ERROR] model path not found: {args.model}")
        return 2
    if not torch.cuda.is_available():
        print("[ERROR] CUDA/ROCm device unavailable.")
        return 2

    baseline = _run_leg(
        model=args.model,
        prompt=args.prompt,
        token_start=args.token_start,
        token_end=args.token_end,
        cos_threshold=args.cos_threshold,
        kv_type=args.kv_type,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        guard_enabled=False,
        guard_start=None,
        guard_span=None,
    )
    drift_layer = None
    if baseline["ok"] and isinstance(baseline["report"], dict):
        drift_layer = baseline["report"].get("first_drift_layer")
    guard_start = int(drift_layer) if drift_layer is not None else 8
    guarded = _run_leg(
        model=args.model,
        prompt=args.prompt,
        token_start=args.token_start,
        token_end=args.token_end,
        cos_threshold=args.cos_threshold,
        kv_type=args.kv_type,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        guard_enabled=True,
        guard_start=guard_start,
        guard_span=max(1, int(args.guard_span)),
    )

    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "token_start": args.token_start,
        "token_end": args.token_end,
        "cos_threshold": args.cos_threshold,
        "kv_type": args.kv_type,
        "guard": {
            "start": guard_start,
            "span": max(1, int(args.guard_span)),
        },
        "baseline": baseline,
        "guarded": guarded,
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None)
    print(text)
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
        print(f"[Saved] {out_path}")

    if not baseline["ok"] and not guarded["ok"]:
        return 2
    if not baseline["ok"] or not guarded["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
