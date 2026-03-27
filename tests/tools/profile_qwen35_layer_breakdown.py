# SPDX-License-Identifier: Apache-2.0
"""
Profile Qwen3.5 (LiteEngine) decode: GPU time share for full-attention block vs linear_attn
submodule vs MLP.

Primary method: torch.cuda.Event forward hooks (works on CUDA and ROCm/HIP). PyTorch profiler
module tags often report 0 GPU time on ROCm; hooks measure real kernel overlap per submodule.

Usage (from repo root, with model on disk):
  uv run python tests/tools/profile_qwen35_layer_breakdown.py \\
    --model models/Qwen3.5-9B-AWQ --decode-steps 16 --warmup-decode 4

  # First prefill chunk only (compare vs decode token-by-token cost):
  uv run python tests/tools/profile_qwen35_layer_breakdown.py \\
    --phase prefill --prefill-chunks 1 --prompt-tokens 512

  # Both: profile first prefill chunk(s), then finish prefill, warmup decode, profile decode
  uv run python tests/tools/profile_qwen35_layer_breakdown.py --phase both

Optional:
  --torch-profiler   Also run torch.profiler (NVIDIA-friendly) + top ops table
  --chrome-trace PATH   Export Chrome trace (use with --torch-profiler)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch

from vllm import SamplingParams
from vllm.config import CacheConfig, LoadConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.engine.lite_engine import LiteEngine
from vllm.model_executor.model_loader import get_tokenizer
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5FullAttentionLayer,
    Qwen3_5LinearAttentionLayer,
)


def _read_awq_group_size_and_bits(model_path: str) -> tuple[int, int]:
    config_path = os.path.join(model_path, "config.json")
    group_size, bits = 128, 4
    try:
        if not os.path.isfile(config_path):
            return group_size, bits
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        qc = raw.get("quantization_config") or {}
        groups = qc.get("config_groups")
        if isinstance(groups, dict):
            for g in groups.values():
                if not isinstance(g, dict):
                    continue
                w = g.get("weights")
                if isinstance(w, dict):
                    if w.get("group_size") is not None:
                        group_size = int(w["group_size"])
                    if w.get("num_bits") is not None:
                        bits = int(w["num_bits"])
                    break
        if qc.get("group_size") is not None:
            group_size = int(qc["group_size"])
        if qc.get("bits") is not None:
            bits = int(qc["bits"])
    except Exception:
        pass
    return group_size, bits


def _build_prompt(tokenizer, target_tokens: int) -> str:
    sentence = (
        "Please explain how modern AI systems improve software performance and reliability "
        "in practical engineering workflows. "
    )
    target_tokens = max(8, int(target_tokens))
    repeat = max(8, target_tokens // 12)
    prompt_text = sentence * repeat
    token_ids = tokenizer.encode(prompt_text)
    if len(token_ids) < target_tokens:
        while len(token_ids) < target_tokens:
            prompt_text = prompt_text + sentence
            token_ids = tokenizer.encode(prompt_text)
    elif len(token_ids) > target_tokens:
        token_ids = token_ids[:target_tokens]
        prompt_text = tokenizer.decode(token_ids)
    return prompt_text


def _event_module_str(event) -> str:
    mh = getattr(event, "module_hierarchy", None)
    if mh is not None:
        return str(mh)
    mod = getattr(event, "module", None)
    if mod is not None:
        return str(mod)
    return ""


def _bucket_for_module(module_str: str) -> str:
    s = module_str
    if not s:
        return "other"
    if "self_attn" in s:
        return "full_attn_self_attn"
    if "linear_attn" in s:
        return "linear_attn"
    if "mlp" in s or "Moe" in s or "moe" in s:
        return "mlp_or_moe_ffn"
    if "embed_tokens" in s or "lm_head" in s or "model.norm" in s or ".norm" in s:
        return "embed_lm_norm"
    return "other"


def _device_time_us(event) -> float:
    for attr in ("device_time_total", "cuda_time_total"):
        v = getattr(event, attr, None)
        if v is not None and float(v) > 0:
            return float(v)
    return 0.0


def _bucket_cuda_time(prof: torch.profiler.profile) -> Tuple[Dict[str, float], float]:
    buckets: Dict[str, float] = {}
    total_cuda = 0.0
    for e in prof.events():
        cuda_us = _device_time_us(e)
        if cuda_us <= 0:
            continue
        total_cuda += cuda_us
        key = _bucket_for_module(_event_module_str(e))
        buckets[key] = buckets.get(key, 0.0) + cuda_us
    return buckets, total_cuda


class _LayerCudaTimers:
    """
    Per-forward CUDA elapsed (ms) from forward hooks (prefill or decode). Cleared after flush().
    """

    def __init__(self, root: torch.nn.Module):
        self._pairs: List[Tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._handles: List[Any] = []
        self._register(root)

    def _enqueue(self, tag: str, e0: torch.cuda.Event, e1: torch.cuda.Event) -> None:
        self._pairs.append((tag, e0, e1))

    def _wrap(self, module: torch.nn.Module, tag: str) -> None:
        def pre(_m: torch.nn.Module, _inp: object) -> None:
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            _m._fi_ev0, _m._fi_ev1 = e0, e1
            e0.record()

        def post(m: torch.nn.Module, _inp: object, _out: object) -> None:
            m._fi_ev1.record()
            self._enqueue(tag, m._fi_ev0, m._fi_ev1)

        self._handles.append(module.register_forward_pre_hook(pre))
        self._handles.append(module.register_forward_hook(post))

    def _register(self, root: torch.nn.Module) -> None:
        inner = getattr(root, "model", None)
        if inner is None:
            return
        if hasattr(inner, "embed_tokens"):
            self._wrap(inner.embed_tokens, "embed_tokens")
        if hasattr(inner, "norm"):
            self._wrap(inner.norm, "final_norm")
        if hasattr(root, "lm_head"):
            self._wrap(root.lm_head, "lm_head")

        layers = getattr(inner, "layers", None)
        if layers is None:
            return
        for li, layer in enumerate(layers):
            if isinstance(layer, Qwen3_5FullAttentionLayer):
                self._wrap(layer, f"layer_full_{li}")
                self._wrap_mlp_linears(layer.mlp, "full", li)
            elif isinstance(layer, Qwen3_5LinearAttentionLayer):
                self._wrap(layer, f"layer_linear_{li}")
                self._wrap_mlp_linears(layer.mlp, "linear", li)

    def _wrap_mlp_linears(self, mlp_mod: torch.nn.Module, kind: str, li: int) -> None:
        # Dense FFN: no single mlp.forward(); time gate/up/down LiteLinear calls.
        for name in ("gate_proj", "up_proj", "down_proj"):
            if hasattr(mlp_mod, name):
                self._wrap(getattr(mlp_mod, name), f"mlp_{kind}_{li}_{name}")

    def flush_ms(self) -> Dict[str, float]:
        torch.cuda.synchronize()
        out: Dict[str, float] = defaultdict(float)
        for tag, e0, e1 in self._pairs:
            out[tag] += e0.elapsed_time(e1)
        self._pairs.clear()
        return dict(out)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _summarize_hook_ms(raw: Dict[str, float]) -> Dict[str, float]:
    """Roll per-layer tags into a few buckets (milliseconds per engine.step)."""
    mlp_ms = 0.0
    full_block_ms = 0.0
    embed_ms = raw.get("embed_tokens", 0.0)
    final_norm_ms = raw.get("final_norm", 0.0)
    lm_head_ms = raw.get("lm_head", 0.0)
    sum_layer_ms = 0.0

    for k, v in raw.items():
        if k.startswith("mlp_full_") or k.startswith("mlp_linear_"):
            mlp_ms += v
        elif k.startswith("layer_full_"):
            idx = int(k.split("_")[-1])
            lf = v
            mf = (
                raw.get(f"mlp_full_{idx}_gate_proj", 0.0)
                + raw.get(f"mlp_full_{idx}_up_proj", 0.0)
                + raw.get(f"mlp_full_{idx}_down_proj", 0.0)
            )
            full_block_ms += max(0.0, lf - mf)
        if k.startswith("layer_full_") or k.startswith("layer_linear_"):
            sum_layer_ms += v

    # Linear-layer path excluding MLP: layer - mlp (includes linear_attn + norms + residual adds)
    linear_path_ms = 0.0
    for k, v in raw.items():
        if not k.startswith("layer_linear_"):
            continue
        idx = int(k.split("_")[-1])
        ml = (
            raw.get(f"mlp_linear_{idx}_gate_proj", 0.0)
            + raw.get(f"mlp_linear_{idx}_up_proj", 0.0)
            + raw.get(f"mlp_linear_{idx}_down_proj", 0.0)
        )
        linear_path_ms += max(0.0, v - ml)

    forward_total_ms = embed_ms + final_norm_ms + lm_head_ms + sum_layer_ms

    return {
        "embed_ms": embed_ms,
        "final_norm_ms": final_norm_ms,
        "lm_head_ms": lm_head_ms,
        "mlp_ms": mlp_ms,
        "full_attn_block_ms": full_block_ms,
        "linear_attn_path_ms": linear_path_ms,
        "forward_total_ms": forward_total_ms,
        "decode_total_ms": forward_total_ms,
        "sum_layer_ms": sum_layer_ms,
    }


def _prefill_done(engine: LiteEngine) -> bool:
    if not engine._running_ids:
        return True
    return all(not engine._requests[rid]["is_prefill"] for rid in engine._running_ids)


def _max_steps_for_prompt(engine: LiteEngine, prompt_len: int, max_new: int) -> int:
    chunk_sz = max(1, int(getattr(engine, "_prefill_chunk_size", 512)))
    prefill_chunks = max(1, (prompt_len + chunk_sz - 1) // chunk_sz)
    return prefill_chunks + max_new * 4 + 200


def _profile_engine_steps(
    engine: LiteEngine,
    timers: _LayerCudaTimers,
    max_steps: int,
    *,
    only_prefill: bool,
) -> Tuple[Dict[str, float], int]:
    """Run engine.step() up to max_steps; optionally stop when prefill is finished."""
    acc: Dict[str, float] = defaultdict(float)
    n = 0
    for _ in range(max_steps):
        if not engine._running_ids:
            break
        if only_prefill:
            rid = engine._running_ids[0]
            if not engine._requests[rid]["is_prefill"]:
                break
        engine.step()
        n += 1
        for k, v in timers.flush_ms().items():
            acc[k] += v
    return dict(acc), n


def _print_bucket_summary(
    summary: Dict[str, float],
    *,
    section_title: str,
    profiled_steps: int,
    total_label: str,
) -> None:
    sum_layer = float(summary.get("sum_layer_ms", 0.0))
    full_b = float(summary.get("full_attn_block_ms", 0.0))
    lin_p = float(summary.get("linear_attn_path_ms", 0.0))
    mlp_t = float(summary.get("mlp_ms", 0.0))
    core_sum = full_b + lin_p + mlp_t
    denom_core = max(1e-6, core_sum)
    total_fwd = max(1e-9, float(summary.get("forward_total_ms", summary.get("decode_total_ms", 0.0))))

    print(f"\n=== {section_title} (CUDA events, mean over {profiled_steps} step(s)) ===")
    print("  Layer stack (full block + linear path + MLP matmuls; should ≈ sum of per-layer times):")
    print(
        f"    full_block + linear_path + mlp  = {core_sum:.3f} ms   "
        f"(sum of layer_* timers = {sum_layer:.3f} ms, "
        f"Δ = {abs(core_sum - sum_layer):.3f} ms)"
    )
    if sum_layer > 1.0 and abs(core_sum - sum_layer) / sum_layer > 0.08:
        print(
            "    [Warn] Mismatch >8%: nested CUDA events on some stacks (e.g. ROCm) — treat as approximate."
        )

    order_keys = [
        ("full_attn_block_ms", "full-attn block (full layers: layer − gate/up/down)"),
        ("linear_attn_path_ms", "linear-attn path (linear layers: layer − gate/up/down)"),
        ("mlp_ms", "MLP matmuls (gate+up+down, all layers)"),
    ]
    for key, label in order_keys:
        ms = float(summary.get(key, 0.0))
        print(f"  {label:55s}  {ms:8.3f} ms  ({100.0 * ms / denom_core:5.1f}% of layer stack)")
    print()
    for key, label in [
        ("embed_ms", "embed_tokens"),
        ("final_norm_ms", "final RMSNorm"),
        ("lm_head_ms", "lm_head"),
    ]:
        ms = float(summary.get(key, 0.0))
        print(f"  {label:55s}  {ms:8.3f} ms  ({100.0 * ms / total_fwd:5.1f}% of {total_label})")
    print(f"  {total_label:55s}  {total_fwd:8.3f} ms")


def _print_notes() -> None:
    print(
        "\nNotes:\n"
        "  • full-attn block: full-attention layer forward minus its MLP matmuls (QKV, rotary, KV, paged attn, o_proj, norms).\n"
        "  • linear-attn path: linear-attention layer forward minus gate/up/down — conv1d, gated delta, out_proj, norms.\n"
        "  • MLP time sums gate_proj, up_proj, down_proj hooks (dense 9B; not MoE).\n"
        "  • Prefill: first chunk(s) use seq_len>1; full-attn may use eager causal prefill; costs differ from decode."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.5 LiteEngine layer profiler (prefill vs decode).")
    parser.add_argument(
        "--model",
        type=str,
        default="models/Qwen3.5-9B-AWQ",
        help="Model directory (same as LiteEngine).",
    )
    parser.add_argument("--prompt-tokens", type=int, default=128, help="Target prompt length (tokens).")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument(
        "--phase",
        type=str,
        choices=("decode", "prefill", "both"),
        default="decode",
        help="decode: after prefill, profile decode. prefill: profile first N prefill chunks only. both: prefill first then decode.",
    )
    parser.add_argument(
        "--prefill-chunks",
        type=int,
        default=1,
        help="Number of consecutive prefill chunks to profile from the start (first chunk = first step). Ignored for --phase decode.",
    )
    parser.add_argument(
        "--warmup-decode",
        type=int,
        default=4,
        help="Decode steps before profiling (not recorded). Used for --phase decode and both.",
    )
    parser.add_argument(
        "--decode-steps",
        type=int,
        default=24,
        help="Profiled decode steps (engine.step each). Used for --phase decode and both.",
    )
    parser.add_argument(
        "--torch-profiler",
        action="store_true",
        help="Also run torch.profiler (often useful on NVIDIA; ROCm may show CPU-only ops).",
    )
    parser.add_argument("--chrome-trace", type=str, default="", help="Optional path to export Chrome trace JSON (requires --torch-profiler).")
    parser.add_argument(
        "--stable-env",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help="Extra env (repeatable), e.g. FASTINFERENCE_KV_FP8=1",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required for this profiler.", file=sys.stderr)
        sys.exit(2)
    if not os.path.isdir(args.model):
        print(f"Model path not found: {args.model}", file=sys.stderr)
        sys.exit(2)

    for item in args.stable_env:
        if "=" in item:
            k, v = item.split("=", 1)
            os.environ[k.strip()] = v.strip()

    os.environ.setdefault("FASTINFERENCE_KV_FP8", "1")

    from vllm.model_executor.layers.quantization.awq import AWQConfig

    gs, wb = _read_awq_group_size_and_bits(args.model)
    m_cfg = ModelConfig(
        model=args.model,
        tokenizer=args.model,
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="float16",
        max_model_len=args.max_model_len,
    )
    c_cfg = CacheConfig(block_size=16, gpu_memory_utilization=args.gpu_memory_utilization, swap_space=0)
    s_cfg = SchedulerConfig(
        max_num_batched_tokens=min(8192, args.max_model_len * 8),
        max_num_seqs=8,
        max_model_len=args.max_model_len,
    )
    q_cfg = AWQConfig(weight_bits=wb, group_size=gs)
    v_cfg = VllmConfig(m_cfg, c_cfg, s_cfg, LoadConfig(load_format="auto"), quant_config=q_cfg)

    print(f"[1/4] Loading LiteEngine model={args.model} AWQ bits={wb} group_size={gs}")
    engine = LiteEngine(v_cfg)
    # LiteEngine leaves tokenizer unset until async loaders wire it; set explicitly for add_request().
    engine.tokenizer = get_tokenizer(args.model, trust_remote_code=True)
    tokenizer = engine.tokenizer
    prompt = _build_prompt(tokenizer, min(args.prompt_tokens, args.max_model_len - 32))
    tok_len = len(tokenizer.encode(prompt))
    print(f"[2/4] Prompt tokens={tok_len} (target ~{args.prompt_tokens})")

    if args.prefill_chunks < 1:
        print("--prefill-chunks must be >= 1", file=sys.stderr)
        sys.exit(2)

    chunk_sz = max(1, int(getattr(engine, "_prefill_chunk_size", 512)))
    if args.phase == "prefill":
        max_tok = max(32, args.prefill_chunks + 16)
    else:
        max_tok = max(64, args.warmup_decode + args.decode_steps + 8)
    sp = SamplingParams(max_tokens=max_tok, temperature=0.0)
    engine.add_request("p0", prompt, sp)

    def _emit_torch_profiler(prof: Optional[torch.profiler.profile], tag: str) -> None:
        if prof is None:
            return
        buckets, total_cuda = _bucket_cuda_time(prof)
        print(f"\n=== torch.profiler [{tag}]: bucketed device time (module tags, if available) ===")
        if total_cuda <= 0:
            print("  No device_time_total on events (common on ROCm); rely on CUDA hooks above.")
        else:
            for name, us in sorted(buckets.items(), key=lambda kv: -kv[1]):
                pct = 100.0 * us / total_cuda
                print(f"  {name:28s}  {us:12.1f} us  ({pct:5.1f}%)")
        evs = prof.events()
        sort_key = "cuda_time_total"
        if evs and hasattr(evs[0], "device_time_total"):
            sort_key = "device_time_total"
        print(f"\n=== torch.profiler [{tag}]: top ops (sort_by={sort_key}) ===")
        try:
            print(prof.key_averages().table(sort_by=sort_key, row_limit=30))
        except Exception as exc:
            print(f"  (table failed: {exc})")

    prof_prefill: Optional[torch.profiler.profile] = None
    prof_decode: Optional[torch.profiler.profile] = None

    # --- Prefill: first N consecutive prefill chunks from a fresh request ---
    if args.phase in ("prefill", "both"):
        n_chunks = int(args.prefill_chunks)
        print(
            f"[3] Phase prefill: profiling first {n_chunks} prefill chunk(s) "
            f"(engine prefill_chunk_size={chunk_sz}; each step processes up to that many new tokens)"
        )
        timers = _LayerCudaTimers(engine.model)
        acc_p: Dict[str, float] = defaultdict(float)
        steps_p = 0
        try:
            if args.torch_profiler:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=False,
                    with_modules=True,
                    profile_memory=False,
                ) as prof_prefill:
                    acc_p, steps_p = _profile_engine_steps(
                        engine, timers, n_chunks, only_prefill=True
                    )
            else:
                acc_p, steps_p = _profile_engine_steps(
                    engine, timers, n_chunks, only_prefill=True
                )
        finally:
            timers.remove()

        denom_p = max(1, steps_p)
        raw_avg_p = {k: v / denom_p for k, v in acc_p.items()}
        summary_p = _summarize_hook_ms(raw_avg_p)
        _print_bucket_summary(
            summary_p,
            section_title=f"Prefill — first {steps_p} chunk step(s)",
            profiled_steps=steps_p,
            total_label="prefill forward total (embed+layers+norm+lm)",
        )
        _emit_torch_profiler(prof_prefill if args.torch_profiler else None, "prefill")
        if args.torch_profiler and args.chrome_trace and args.phase == "prefill":
            trace_path = os.path.abspath(args.chrome_trace)
            prof_prefill.export_chrome_trace(trace_path)
            print(f"\nChrome trace written: {trace_path}")

        if args.phase == "prefill":
            _print_notes()
            while engine._running_ids:
                engine.step()
            return

    # --- Finish prefill (both + decode), then optional decode warmup + profile ---
    if args.phase in ("decode", "both"):
        max_steps = _max_steps_for_prompt(engine, tok_len, sp.max_tokens or 32)
        steps_run = 0
        while not _prefill_done(engine) and steps_run < max_steps:
            engine.step()
            steps_run += 1
        if not _prefill_done(engine):
            raise RuntimeError("Prefill did not finish within step budget.")

        label = "[3/4]" if args.phase == "decode" else "[4]"
        if steps_run == 0:
            print(
                f"{label} Prefill already complete after profiled chunk(s) "
                f"(0 extra prefill steps); warming up decode x{args.warmup_decode}"
            )
        else:
            print(
                f"{label} Drained remaining prefill in {steps_run} step(s); "
                f"warming up decode x{args.warmup_decode}"
            )
        for _ in range(args.warmup_decode):
            if not engine._running_ids:
                raise RuntimeError("Request finished before warmup decode.")
            engine.step()
            steps_run += 1

        torch.cuda.synchronize()
        print(
            f"{label} Profiling decode: {args.decode_steps} step(s) (CUDA event hooks + optional torch.profiler)"
        )

        timers = _LayerCudaTimers(engine.model)
        acc: Dict[str, float] = defaultdict(float)
        profiled_decode_steps = 0
        try:

            def _run_profiled_decode() -> None:
                nonlocal profiled_decode_steps
                for _ in range(args.decode_steps):
                    if not engine._running_ids:
                        break
                    engine.step()
                    profiled_decode_steps += 1
                    for k, v in timers.flush_ms().items():
                        acc[k] += v

            if args.torch_profiler:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=False,
                    with_modules=True,
                    profile_memory=False,
                ) as prof_decode:
                    _run_profiled_decode()
            else:
                _run_profiled_decode()
        finally:
            timers.remove()

        denom_steps = max(1, profiled_decode_steps)
        raw_avg = {k: v / denom_steps for k, v in acc.items()}
        summary = _summarize_hook_ms(raw_avg)
        _print_bucket_summary(
            summary,
            section_title="Decode",
            profiled_steps=profiled_decode_steps,
            total_label="decode forward total (embed+layers+norm+lm)",
        )
        _emit_torch_profiler(prof_decode if args.torch_profiler else None, "decode")
        if args.torch_profiler and args.chrome_trace and args.phase in ("decode", "both"):
            trace_path = os.path.abspath(args.chrome_trace)
            prof_decode.export_chrome_trace(trace_path)
            print(f"\nChrome trace written (decode): {trace_path}")

    _print_notes()

    # Clean up running request if any
    while engine._running_ids:
        engine.step()


if __name__ == "__main__":
    main()
