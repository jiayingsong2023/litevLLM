#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Compare HF vs Lite hidden states for Qwen3.5 dense checkpoints.

Primary hook: output of final RMSNorm (``model.norm``), i.e. the tensor fed into ``lm_head``.
This matches HF ``Qwen3_5TextModel`` last hidden state (post-final-norm).

Optional: per-layer submodule hooks on ``model.layers.{i}`` outputs to see where drift starts.

Requires the same prompt tokenization as other audits; uses LiteEngine prefill path for Lite.

Usage:
  PYTHONPATH=. uv run python tests/tools/verify_qwen35_final_hidden_alignment.py \\
    --model models/Qwen3.5-9B-FP16 --hf-model models/Qwen3.5-9B-FP16 --quant none

  # Optional: also print CosSim for each layer output (last token position)
  PYTHONPATH=. uv run python tests/tools/verify_qwen35_final_hidden_alignment.py \\
    --model models/Qwen3.5-9B-FP16 --hf-model models/Qwen3.5-9B-FP16 --quant none --per-layer
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("FASTINFERENCE_KV_TYPE", "turbo_int4")

import transformers.utils.import_utils as _transformers_import_utils

_transformers_import_utils.is_flash_linear_attention_available = lambda: False  # noqa: E731

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from vllm.config import CacheConfig, LoadConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.engine.lite_engine import LiteEngine
from vllm.sampling_params import SamplingParams


def _pair_metrics(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    x = a.float().reshape(-1)
    y = b.float().reshape(-1)
    n = min(x.numel(), y.numel())
    x = x[:n]
    y = y[:n]
    mask = torch.isfinite(x) & torch.isfinite(y)
    if not mask.any():
        return {"cos_sim": float("nan"), "mae": float("nan"), "max_err": float("nan")}
    x = x[mask]
    y = y[mask]
    diff = (x - y).abs()
    cos_sim = F.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0), dim=1).squeeze(0).item()
    return {"cos_sim": cos_sim, "mae": diff.mean().item(), "max_err": diff.max().item()}


def _resolve_hf_dtype(hf_model_path: str) -> torch.dtype:
    try:
        p = os.path.join(hf_model_path, "config.json")
        with open(p, "r") as f:
            raw = json.load(f)
        tc = raw.get("text_config") or {}
        ds = tc.get("dtype") or tc.get("torch_dtype")
        if ds is None:
            ds = raw.get("torch_dtype")
        if isinstance(ds, str) and "bfloat16" in ds.lower():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


def _build_quant_config(quant: str, model_path: str):
    if quant == "none":
        return None
    if quant == "awq":
        from vllm.model_executor.layers.quantization.awq import AWQConfig

        cfg_path = os.path.join(model_path, "config.json")
        group_size, bits = 128, 4
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r") as f:
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
        return AWQConfig(weight_bits=bits, group_size=group_size)
    if quant == "gguf":
        from vllm.model_executor.layers.quantization.gguf import GGUFConfig

        return GGUFConfig()
    raise ValueError(f"Unknown quant: {quant}")


def _register_output_hooks(
    root: torch.nn.Module, submodule_paths: List[str], store: Dict[str, torch.Tensor]
) -> List:
    modules = dict(root.named_modules())
    handles = []
    for path in submodule_paths:
        mod = modules.get(path)
        if mod is None:
            sys.stderr.write(f"[Warning] Missing module path: {path!r}\n")
            continue

        def _make_hook(key: str):
            def _hook(_m, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                if torch.is_tensor(t):
                    store[key] = t.detach()

            return _hook

        handles.append(mod.register_forward_hook(_make_hook(path)))
    return handles


def _hf_layer_paths(num_layers: int) -> List[str]:
    return [f"model.layers.{i}" for i in range(num_layers)]


def main() -> int:
    parser = argparse.ArgumentParser(description="HF vs Lite Qwen3.5 final hidden (post-norm) alignment.")
    parser.add_argument("--model", type=str, required=True, help="Lite checkpoint directory")
    parser.add_argument("--hf-model", type=str, required=True, help="HF reference directory")
    parser.add_argument("--quant", type=str, default="none", choices=["none", "awq", "gguf"])
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument(
        "--hf-device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for HF forward (cuda reduces CPU-vs-GPU kernel drift vs Lite on CUDA).",
    )
    parser.add_argument(
        "--per-layer",
        action="store_true",
        help="Also hook each model.layers.{i} output and report CosSim vs HF (last token).",
    )
    args = parser.parse_args()

    m_cfg = ModelConfig(model=args.model, tokenizer=args.model)
    c_cfg = CacheConfig(block_size=16, gpu_memory_utilization=0.9, swap_space=4)
    s_cfg = SchedulerConfig(max_num_batched_tokens=2048, max_num_seqs=32, max_model_len=2048)
    l_cfg = LoadConfig()
    q_cfg = _build_quant_config(args.quant, args.model)
    v_cfg = VllmConfig(m_cfg, c_cfg, s_cfg, l_cfg, quant_config=q_cfg)

    from vllm.model_executor.model_loader import get_tokenizer

    tokenizer = get_tokenizer(args.hf_model, trust_remote_code=True)
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt")
    print(f"[Input] token_count={input_ids.shape[1]} ids={input_ids[0].tolist()}")

    hf_dtype = _resolve_hf_dtype(args.hf_model)
    hf_cfg = AutoConfig.from_pretrained(args.hf_model, trust_remote_code=True)
    hf_kw = dict(
        pretrained_model_name_or_path=args.hf_model,
        config=hf_cfg,
        trust_remote_code=True,
        dtype=hf_dtype,
        low_cpu_mem_usage=True,
    )
    if getattr(hf_cfg, "model_type", "") == "qwen3_5":
        hf_kw["attn_implementation"] = "eager"
    hf_model = AutoModelForCausalLM.from_pretrained(**hf_kw).eval()

    num_layers = int(getattr(hf_cfg, "num_hidden_layers", 0) or getattr(hf_model.config, "num_hidden_layers", 0))
    if num_layers <= 0:
        num_layers = len(hf_model.model.layers)

    hf_dev = torch.device(args.hf_device if args.hf_device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.hf_device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA unavailable for HF; using CPU.")
        hf_dev = torch.device("cpu")

    hf_layer_hook_paths = _hf_layer_paths(num_layers) if args.per_layer else []

    hf_model = hf_model.to(hf_dev)
    hf_store: Dict[str, torch.Tensor] = {}
    hf_handles: List = []
    if hf_layer_hook_paths:
        hf_handles = _register_output_hooks(hf_model, hf_layer_hook_paths, hf_store)
    attn_mask = torch.ones_like(input_ids, dtype=torch.long, device=hf_dev)
    with torch.inference_mode():
        hf_text_out = hf_model.model(
            input_ids=input_ids.to(hf_dev),
            attention_mask=attn_mask,
            use_cache=False,
        )
    for h in hf_handles:
        h.remove()

    # last_hidden_state is post-final-norm (see transformers Qwen3_5TextModel.forward)
    hf_norm = hf_text_out.last_hidden_state
    if hf_norm is None:
        print("[Error] HF model returned no last_hidden_state.")
        return 1
    hf_last = hf_norm[:, -1, :].float().cpu()

    print("\n[HF] Post-final-norm hidden (last token), from hf_model.model(...).last_hidden_state:")
    print(f"  shape={tuple(hf_norm.shape)}")

    # Lite: LiteEngine prefill (same path as production)
    print("\n[Lite] Loading LiteEngine...")
    engine = LiteEngine(v_cfg)
    engine.tokenizer = tokenizer
    lite_root = engine.model

    lite_hook_paths = (
        _hf_layer_paths(num_layers) + ["model.norm"] if args.per_layer else ["model.norm"]
    )

    lite_store: Dict[str, torch.Tensor] = {}
    lite_handles = _register_output_hooks(lite_root, lite_hook_paths, lite_store)

    # Run one prefill chunk + one greedy token (hook fires during prefill)
    engine.add_request("hidden_audit", args.prompt, SamplingParams(max_tokens=1, temperature=0.0))
    step_budget = 2048
    for _ in range(step_budget):
        outs = engine.step()
        if outs:
            break
    else:
        for h in lite_handles:
            h.remove()
        print("[Error] LiteEngine step did not return output.")
        return 1

    for h in lite_handles:
        h.remove()

    lite_norm = lite_store.get("model.norm")
    if lite_norm is None:
        print("[Error] Lite did not capture model.norm output.")
        return 1

    lite_last = lite_norm[:, -1, :].float().cpu()
    if hf_norm.shape != lite_norm.shape:
        print(
            f"[Note] shape mismatch HF={tuple(hf_norm.shape)} Lite={tuple(lite_norm.shape)}; "
            "metrics use flattened min length where needed."
        )

    pm = _pair_metrics(lite_last, hf_last)
    print("\n[Compare] Lite vs HF — post-final-norm hidden, last token position:")
    print(
        f"  CosSim={pm['cos_sim']:.6f}  MAE={pm['mae']:.6f}  MaxErr={pm['max_err']:.6f}"
    )

    if pm["cos_sim"] < 0.95:
        print(
            "\n[Interpret] Low CosSim is usually NOT fixed by RoPE alone on plain text:\n"
            "  • Lite applies Qwen3.5 MRoPE cos/sin consistent with HF (see MRotaryEmbedding\n"
            "    _forward_hf_mrope); for text-only prompts, T/H/W rows are identical so the old\n"
            "    1D cos_cached indexing already matched HF cos per position.\n"
            "  • Remaining drift is dominated by linear-attn (GatedDeltaNet) vs HF and full-attn\n"
            "    paged path vs HF eager — use this script’s per-layer table to localize.\n"
            "  • The 3D position_ids path is still required for correctness when multimodal rows\n"
            "    differ or for future parity tests against HF."
        )

    # Logits sanity: lm_head(h_norm) vs HF logits last position
    with torch.inference_mode():
        w = hf_model.lm_head.weight.float().cpu()
        hf_logits_last = F.linear(hf_last, w)
        lw = lite_root.lm_head.weight.float().cpu()
        lite_logits_last = F.linear(lite_last, lw)
    tok_hf = int(torch.argmax(hf_logits_last).item())
    tok_lite = int(torch.argmax(lite_logits_last).item())
    print("\n[Argmax] Greedy next token from post-norm hidden × lm_head (last position):")
    print(f"  HF={tok_hf}  Lite={tok_lite}  (match={tok_hf == tok_lite})")

    if args.per_layer:
        print("\n[Per-layer] CosSim at last token (Lite vs HF layer output):")
        drift_at: Optional[int] = None
        for i in range(num_layers):
            key = f"model.layers.{i}"
            h_t = hf_store.get(key)
            l_t = lite_store.get(key)
            if h_t is None or l_t is None:
                print(f"  layer {i}: missing (hf={h_t is not None}, lite={l_t is not None})")
                continue
            hl = h_t[:, -1, :].float().cpu()
            ll = l_t[:, -1, :].float().cpu()
            pm_i = _pair_metrics(ll, hl)
            flag = "  "
            # Heuristic: first layer where stream is clearly off HF (skip "almost aligned" ~0.9x band)
            if drift_at is None and pm_i["cos_sim"] < 0.85:
                drift_at = i
                flag = "* "
            print(
                f"{flag}layer {i:2d}: CosSim={pm_i['cos_sim']:.6f}  "
                f"MAE={pm_i['mae']:.6f}  MaxErr={pm_i['max_err']:.6f}"
            )
        if drift_at is not None:
            print(
                f"\n[Hint] First layer with CosSim < 0.85 at last token: {drift_at} "
                f"(inspect full-attn vs linear-attn pattern every 4 layers in config)."
            )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
