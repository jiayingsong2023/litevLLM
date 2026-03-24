#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
MoE GGUF packed path: audits for Lite logits vs dense load and for raw weight parity.

## A) Last-prefill logits (full Lite stack, two runs)

Compare Lite prefill logits when loading the same GGUF with packed MoE vs dense MoE experts.
Use two processes (dense load often needs more host RAM).

  FASTINFERENCE_QWEN35_MOE_PACKED_GGUF=1 uv run python tests/tools/qwen35_moe_packed_lite_logits_audit.py dump \\
    --model models/Qwen3.5-35B-MoE-GGUF --out /tmp/lite_packed.pt

  FASTINFERENCE_QWEN35_MOE_PACKED_GGUF=0 uv run python tests/tools/qwen35_moe_packed_lite_logits_audit.py dump \\
    --model models/Qwen3.5-35B-MoE-GGUF --out /tmp/lite_dense.pt --frugal

  uv run python tests/tools/qwen35_moe_packed_lite_logits_audit.py diff /tmp/lite_packed.pt /tmp/lite_dense.pt

## C) Offline stats (no second run)

  uv run python tests/tools/qwen35_moe_packed_lite_logits_audit.py stats /tmp/lite_packed.pt

Prints finite ratio, NaN/Inf counts, mean/std, argmax, top-5 logits (sanity without HF).

High CosSim + low MaxAbs on logits_last => packed MoE weights do not change the forward vs dense load
(for the same prompt); if this fails, suspect loader/slice bugs. If it passes but HF differs, compare HF separately.

## B) Weight-only parity (no Lite model; fast)

Verifies slice dequant from packed bytes matches full-tensor dequant for one expert:

  uv run python tests/tools/qwen35_moe_packed_lite_logits_audit.py weight-parity \\
    --gguf models/Qwen3.5-35B-MoE-GGUF/Qwen3.5-35B-A3B-UD-Q4_K_M.gguf --layer 0 --expert 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("FASTINFERENCE_KV_FP8", "0")


def summarize_logits_last_1d(logits_last: Any) -> Dict[str, Any]:
    """Sanity stats for last-position prefill logits (1D vocab vector). Testable without Lite."""
    import torch

    x = logits_last.detach().float().cpu().flatten()
    n = int(x.numel())
    n_nan = int(torch.isnan(x).sum().item())
    n_inf = int(torch.isinf(x).sum().item())
    finite = torch.isfinite(x)
    n_fin = int(finite.sum().item())
    out: Dict[str, Any] = {
        "numel": n,
        "finite_count": n_fin,
        "finite_ratio": n_fin / max(n, 1),
        "nan_count": n_nan,
        "inf_count": n_inf,
    }
    if n_fin == 0:
        out["argmax"] = -1
        out["mean"] = float("nan")
        out["std"] = float("nan")
        out["min"] = float("nan")
        out["max"] = float("nan")
        out["top5_values"] = []
        out["top5_indices"] = []
        return out
    xv = x[finite]
    out["mean"] = float(xv.mean().item())
    out["std"] = float(xv.std().item()) if xv.numel() > 1 else 0.0
    out["min"] = float(xv.min().item())
    out["max"] = float(xv.max().item())
    neg_inf = torch.tensor(float("-inf"), dtype=x.dtype)
    repl = torch.where(finite, x, neg_inf)
    out["argmax"] = int(torch.argmax(repl).item())
    k = min(5, n)
    vals, idx = torch.topk(repl, k=k)
    out["top5_values"] = [float(v) for v in vals.tolist()]
    out["top5_indices"] = [int(i) for i in idx.tolist()]
    return out


def print_logits_last_summary(
    logits_last: Any,
    payload_meta: Optional[Dict[str, Any]] = None,
) -> None:
    s = summarize_logits_last_1d(logits_last)
    print("--- logits_last stats ---")
    print(f"  numel={s['numel']} finite_ratio={s['finite_ratio']:.6f} "
          f"nan={s['nan_count']} inf={s['inf_count']}")
    print(f"  mean={s['mean']:.6g} std={s['std']:.6g} min={s['min']:.6g} max={s['max']:.6g}")
    print(f"  argmax_id={s['argmax']}")
    print(f"  top5_values={s['top5_values']}")
    print(f"  top5_indices={s['top5_indices']}")
    if payload_meta:
        for key in ("prompt_len", "moe_packed_gguf", "model_path", "prompt"):
            if key in payload_meta and payload_meta[key] is not None:
                pv = payload_meta[key]
                if key == "prompt" and isinstance(pv, str) and len(pv) > 120:
                    pv = pv[:117] + "..."
                print(f"  {key}: {pv}")


def _run_lite_steps_until(
    engine: Any,
    max_steps: int,
    stop_fn: Any,
) -> Any:
    for _ in range(max_steps):
        step_outputs = engine.step()
        done = stop_fn(step_outputs)
        if done is not None:
            return done
    raise RuntimeError(
        f"exceeded {max_steps} LiteEngine.step() calls (active={getattr(engine, 'active_request_count', '?')})"
    )


def _dump_logits(model_dir: str, prompt: str, out_path: str, *, frugal: bool, gpu_mu: Optional[float]) -> None:
    import torch
    from vllm.config import CacheConfig, LoadConfig, ModelConfig, SchedulerConfig, VllmConfig
    from vllm.engine.lite_engine import LiteEngine
    from vllm.model_executor.layers.quantization.gguf import GGUFConfig
    from vllm.sampling_params import SamplingParams

    mu = gpu_mu if gpu_mu is not None else (0.55 if frugal else 0.9)
    max_len = 1024 if frugal else 2048
    max_seqs = 4 if frugal else 32
    mbt = min(8192, max(512, max_len * 4))

    m_cfg = ModelConfig(model=model_dir, tokenizer=model_dir)
    c_cfg = CacheConfig(block_size=16, gpu_memory_utilization=mu, swap_space=4)
    s_cfg = SchedulerConfig(
        max_num_batched_tokens=mbt,
        max_num_seqs=max_seqs,
        max_model_len=max_len,
    )
    l_cfg = LoadConfig()
    q_cfg = GGUFConfig()
    v_cfg = VllmConfig(m_cfg, c_cfg, s_cfg, l_cfg, quant_config=q_cfg)

    engine = LiteEngine(v_cfg)
    from vllm.model_executor.model_loader import get_tokenizer

    tokenizer = get_tokenizer(model_dir, trust_remote_code=True)
    engine.tokenizer = tokenizer

    captured: List[torch.Tensor] = []
    orig = engine.model.forward

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        r = orig(*args, **kwargs)
        captured.append(r.detach().float().cpu())
        return r

    engine.model.forward = wrapped  # type: ignore[assignment]

    prompt_len = len(tokenizer.encode(prompt))
    engine.add_request(
        "audit",
        prompt,
        SamplingParams(max_tokens=1, temperature=0.0),
    )

    chunk_sz = max(1, int(getattr(engine, "_prefill_chunk_size", 512)))
    prefill_chunks = max(1, (prompt_len + chunk_sz - 1) // chunk_sz)
    budget = prefill_chunks + 600

    _run_lite_steps_until(
        engine,
        budget,
        lambda outs: outs[0] if outs else None,
    )

    engine.model.forward = orig  # type: ignore[assignment]

    if not captured:
        raise RuntimeError("No logits captured (forward never ran?)")
    last = captured[-1]
    if last.dim() != 3:
        raise RuntimeError(f"Expected logits [B,T,V], got {tuple(last.shape)}")
    logits_last = last[0, -1, :].clone()

    packed = os.environ.get("FASTINFERENCE_QWEN35_MOE_PACKED_GGUF", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    payload = {
        "logits_last": logits_last,
        "prompt": prompt,
        "prompt_len": prompt_len,
        "moe_packed_gguf": packed,
        "model_path": os.path.abspath(model_dir),
    }
    torch.save(payload, out_path)
    print(f"Saved {out_path} (shape={tuple(logits_last.shape)}, packed_env={packed})")
    meta = {k: v for k, v in payload.items() if k != "logits_last"}
    print_logits_last_summary(logits_last, meta)


def _stats_file(path: str) -> None:
    import torch

    d = torch.load(path, map_location="cpu", weights_only=False)
    if "logits_last" not in d:
        raise KeyError(f"Expected key 'logits_last' in {path}")
    logits_last = d["logits_last"]
    meta = {k: v for k, v in d.items() if k != "logits_last"}
    print_logits_last_summary(logits_last, meta)


def _diff(a_path: str, b_path: str) -> None:
    import torch
    import torch.nn.functional as F

    a = torch.load(a_path, map_location="cpu", weights_only=False)
    b = torch.load(b_path, map_location="cpu", weights_only=False)
    la = a["logits_last"].float().flatten()
    lb = b["logits_last"].float().flatten()
    n = min(la.numel(), lb.numel())
    la = la[:n]
    lb = lb[:n]
    mask = torch.isfinite(la) & torch.isfinite(lb)
    if not mask.any():
        print("No finite logits to compare.")
        return
    la = la[mask]
    lb = lb[mask]
    cos = F.cosine_similarity(la.unsqueeze(0), lb.unsqueeze(0), dim=1).item()
    mx = (la - lb).abs().max().item()
    print(f"Compared logits_last: n={la.numel()}")
    print(f"  CosSim: {cos:.8f}")
    print(f"  MaxAbs: {mx:.6f}")
    ta = la.argmax().item()
    tb = lb.argmax().item()
    print(f"  Argmax: {ta} vs {tb} {'(match)' if ta == tb else '(MISMATCH)'}")


def _weight_parity(gguf_path: str, layer: int, expert: int) -> None:
    import numpy as np
    import torch
    import gguf

    from vllm.model_executor import model_loader as ml
    from vllm.model_executor.moe_gguf_packed import (
        dequant_packed_rows_to_fp16,
        numpy_gguf_data_to_packed_2d,
    )

    cfg_dir = os.path.dirname(os.path.abspath(gguf_path))
    cfg_path = os.path.join(cfg_dir, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Need config.json next to GGUF: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    tc = raw.get("text_config") or raw
    E = int(tc.get("num_experts", raw.get("num_experts", 0)) or 0)
    inter = int(tc.get("moe_intermediate_size", 0) or 0)
    H = int(tc.get("hidden_size", 0) or 0)
    if E <= 0 or inter <= 0 or H <= 0:
        raise ValueError(f"Bad MoE dims E={E} inter={inter} H={H} from {cfg_path}")
    if expert < 0 or expert >= E:
        raise ValueError(f"expert must be in [0,{E})")

    reader = gguf.GGUFReader(gguf_path)
    tensor_map = {t.name: t for t in reader.tensors}
    ge_key = f"blk.{layer}.ffn_gate_exps.weight"
    if ge_key not in tensor_map:
        raise KeyError(f"Missing {ge_key} in {gguf_path}")
    ge = tensor_map[ge_key]
    qt = int(ge.tensor_type)
    logical = (E, inter, H)

    dense = ml._dequantize_gguf_tensor(
        ge, "cpu", torch.float16, target_shape=torch.Size(logical)
    )
    if dense is None:
        raise RuntimeError("dense dequant returned None")

    p2d_np = numpy_gguf_data_to_packed_2d(np.asarray(ge.data), logical, qt)
    p2d = torch.from_numpy(np.ascontiguousarray(p2d_np)).to(torch.uint8)
    r0 = expert * inter
    r1 = r0 + inter
    sl = dequant_packed_rows_to_fp16(p2d, r0, r1, H, qt)

    d0 = dense[expert].float()
    s0 = sl.float()
    mx = (d0 - s0).abs().max().item()
    print(f"{ge_key} expert={expert} qtype={qt} max_abs(dense-slice)={mx:.6g}")
    if mx > 1e-2:
        print("  [FAIL] packed slice dequant vs full dequant mismatch (investigate layout/qtype).")
        sys.exit(1)
    print("  [OK] weight parity for gate expert slice.")


def main() -> int:
    p = argparse.ArgumentParser(description="MoE packed: Lite logits dump/diff + weight parity")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump", help="Run one prefill and save last-position logits")
    d.add_argument("--model", required=True, help="Model directory (config + tokenizer + .gguf)")
    d.add_argument("--prompt", default="The capital of France is Paris.")
    d.add_argument("--out", required=True, help="Path to .pt file")
    d.add_argument(
        "--frugal",
        action="store_true",
        help="Lower gpu_memory_utilization and max_model_len (large MoE GGUF).",
    )
    d.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Override CacheConfig gpu_memory_utilization (default: 0.55 if --frugal else 0.9).",
    )

    c = sub.add_parser("diff", help="Compare two dump .pt files (CosSim / MaxAbs / argmax)")
    c.add_argument("a")
    c.add_argument("b")

    s = sub.add_parser("stats", help="Print summarize_logits_last_1d stats from a dump .pt file")
    s.add_argument("path", help="Path to .pt file saved by dump")

    w = sub.add_parser(
        "weight-parity",
        help="No Lite: compare full dequant vs packed-slice dequant for one expert gate tensor",
    )
    w.add_argument("--gguf", required=True, help="Path to .gguf file")
    w.add_argument("--layer", type=int, default=0)
    w.add_argument("--expert", type=int, default=0)

    args = p.parse_args()
    if args.cmd == "dump":
        _dump_logits(
            args.model,
            args.prompt,
            args.out,
            frugal=bool(args.frugal),
            gpu_mu=args.gpu_memory_utilization,
        )
        return 0
    if args.cmd == "diff":
        _diff(args.a, args.b)
        return 0
    if args.cmd == "stats":
        _stats_file(args.path)
        return 0
    if args.cmd == "weight-parity":
        _weight_parity(args.gguf, args.layer, args.expert)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
