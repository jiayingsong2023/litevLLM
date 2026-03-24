#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Layer-wise hidden states (last token): HuggingFace DeepSeek V2 (bf16/fp16) vs FastInference Lite (GGUF).

Paths (same as compare_hf_lite_deepseek_logits.py):
  --lite-model   Directory with config.json + *.gguf; weights via vllm get_model + GGUF mapping
                 ([vllm/model_executor/model_loader/__init__.py]).
  --hf-model     HF reference checkpoint (local dir or Hub id). Default: local models/DeepSeek-V2-Lite-Chat
                 if present, else deepseek-ai/DeepSeek-V2-Lite-Chat.
  Tokenizer      --tokenizer or _resolved_tokenizer_path: sibling Chat dir via resolve_deepseek_hf_chat_dir
                 when not overridden (matches engine).
  FASTINFERENCE_DEEPSEEK_HF_CHAT   Optional override for HF Chat directory (see deepseek_hf_reference.py).
  FASTINFERENCE_GGUF_DEQUANT_FP32 / FASTINFERENCE_GGUF_DEQUANT_FP8   GGUF dequant intermediates (model_loader).

Hooks (symmetric): base model ``model.model`` — embed_tokens output, each DeepseekV2DecoderLayer output,
final RMSNorm output. Last-position vector only; compare cosine / MSE vs HF.

Q4 GGUF vs bf16 HF will diverge early (quantization); use this to see whether mismatch starts at embed
or jumps at a specific layer (implementation vs quant).

Example:
  uv run python tests/tools/compare_hf_lite_deepseek_layer_hiddens.py \\
    --lite-model models/DeepSeek-V2-Lite-GGUF \\
    --hf-model models/DeepSeek-V2-Lite-Chat \\
    --chat-template auto \\
    --cosine-warn 0.995
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import sys
from typing import Any

import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _load_compare_module():
    path = os.path.join(os.path.dirname(__file__), "compare_hf_lite_deepseek_logits.py")
    spec = importlib.util.spec_from_file_location("compare_hf_lite_deepseek_logits", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load compare_hf_lite_deepseek_logits.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _last_token_hidden(module_out: Any) -> torch.Tensor:
    out = module_out[0] if isinstance(module_out, tuple) else module_out
    return out[0, -1, :].detach().float().cpu()


def _register_deepseek_base_hooks(
    base: nn.Module,
    storage: dict[str, torch.Tensor],
) -> list[Any]:
    """Register forward hooks on DeepseekV2Model: embed_tokens, layers[i], norm."""

    def hook_embed(_m: nn.Module, _inp: Any, out: Any) -> None:
        storage["embed"] = _last_token_hidden(out)

    handles: list[Any] = []
    handles.append(base.embed_tokens.register_forward_hook(hook_embed))

    for i, layer in enumerate(base.layers):

        def layer_hook(_m: nn.Module, _inp: Any, out: Any, li: int = i) -> None:
            storage[f"layer_{li:02d}"] = _last_token_hidden(out)

        handles.append(layer.register_forward_hook(layer_hook))

    def hook_norm(_m: nn.Module, _inp: Any, out: Any) -> None:
        storage["norm"] = _last_token_hidden(out)

    handles.append(base.norm.register_forward_hook(hook_norm))
    return handles


def _remove_hooks(handles: list[Any]) -> None:
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


def _ordered_checkpoint_keys(n_layers: int) -> list[str]:
    return ["embed"] + [f"layer_{i:02d}" for i in range(n_layers)] + ["norm"]


def _hidden_metrics(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    x = a.float()
    y = b.float()
    nx = x.norm()
    ny = y.norm()
    cos = float((x @ y) / (nx * ny + 1e-8)) if nx > 0 and ny > 0 else 0.0
    mse = float(((x - y) ** 2).mean().item())
    mae = float((x - y).abs().mean().item())
    return cos, mse, mae


def _torch_load(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (TypeError, Exception):
        return torch.load(path, map_location="cpu")


def _run_hf_capture(
    cmp: Any,
    hf_model: str,
    input_ids: torch.Tensor,
    dtype: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """One CausalLM forward: hooks on model.model + logits from the same pass."""
    m = cmp._load_hf_causal_lm(hf_model, dtype)
    m.eval()
    base = m.model
    n_layers = len(base.layers)
    storage: dict[str, torch.Tensor] = {}
    handles = _register_deepseek_base_hooks(base, storage)
    ids = input_ids.to(device="cuda", dtype=torch.long)
    seq = ids.shape[1]
    position_ids = torch.arange(seq, device=ids.device, dtype=torch.long).unsqueeze(0)
    try:
        with torch.inference_mode():
            out = m(
                input_ids=ids,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
            )
            logits_last = out.logits[0, -1, :].float().cpu()
    finally:
        _remove_hooks(handles)
    expected = set(_ordered_checkpoint_keys(n_layers))
    missing = expected - set(storage.keys())
    if missing:
        raise RuntimeError(f"HF hooks missing keys: {missing}")
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return storage, logits_last


def _run_lite_capture(
    cmp: Any,
    lite_model: str,
    input_ids: torch.Tensor,
    max_model_len: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    model = cmp.load_lite_model(lite_model, max_model_len)
    model.eval()
    base = model.model
    n_layers = len(base.layers)
    storage: dict[str, torch.Tensor] = {}
    handles = _register_deepseek_base_hooks(base, storage)
    ids = input_ids.to(device="cuda", dtype=torch.long)
    seq = ids.shape[1]
    positions = torch.arange(seq, device=ids.device, dtype=torch.long).unsqueeze(0)
    kv_caches = [None] * n_layers
    meta = cmp._build_lite_attn_metadata(seq, n_layers, max_model_len, ids.device)
    try:
        with torch.inference_mode():
            logits = model(ids, positions, kv_caches, meta, lora_mapping=None)
            logits_last = logits[0, -1, :].float().cpu()
    finally:
        _remove_hooks(handles)
    expected = set(_ordered_checkpoint_keys(n_layers))
    missing = expected - set(storage.keys())
    if missing:
        raise RuntimeError(f"Lite hooks missing keys: {missing}")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return storage, logits_last


def _print_path_info(
    lite_model: str,
    hf_model: str,
    tokenizer_path: str,
) -> None:
    print("[Paths] --lite-model (GGUF tree: config.json + *.gguf):")
    print(f"        {lite_model}")
    print("[Paths] --hf-model (HF reference weights):")
    print(f"        {hf_model}")
    print("[Paths] tokenizer (resolved; sibling Chat dir when not --tokenizer):")
    print(f"        {tokenizer_path}")
    env_chat = os.environ.get("FASTINFERENCE_DEEPSEEK_HF_CHAT", "").strip()
    if env_chat:
        print(f"[Paths] FASTINFERENCE_DEEPSEEK_HF_CHAT={env_chat}")
    print(
        "[Paths] Optional GGUF load: FASTINFERENCE_GGUF_DEQUANT_FP32, "
        "FASTINFERENCE_GGUF_DEQUANT_FP8 (see vllm/model_executor/model_loader/__init__.py)"
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="HF vs Lite: per-layer hidden states (last token) for DeepSeek V2"
    )
    p.add_argument("--lite-model", type=str, required=True)
    p.add_argument("--hf-model", type=str, default=None)
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--prompt", type=str, default="What is 2+2? Answer with one digit.")
    p.add_argument("--chat-template", choices=("off", "auto", "on"), default="off")
    p.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument(
        "--cosine-warn",
        type=float,
        default=0.995,
        help="Report first checkpoint with cosine below this value",
    )
    p.add_argument(
        "--save-hf-hiddens",
        type=str,
        default=None,
        help="Save HF checkpoints + logits_last to this .pt (dict with keys checkpoints, logits_last)",
    )
    p.add_argument(
        "--load-hf-hiddens",
        type=str,
        default=None,
        help="Load HF data from .pt (from --save-hf-hiddens) instead of running HF",
    )
    p.add_argument(
        "--skip-lite",
        action="store_true",
        help="Only run HF and optional --save-hf-hiddens",
    )
    args = p.parse_args()

    cmp = _load_compare_module()
    _apply_chat = cmp._apply_chat_template_if_needed
    _encode = cmp._encode
    _resolved_tok = cmp._resolved_tokenizer_path
    _default_hf = cmp._default_hf_model
    _topk_diff = cmp._topk_diff

    hf_model = args.hf_model or _default_hf()
    tokenizer_path = _resolved_tok(args.lite_model, args.tokenizer)

    try:
        from vllm.model_executor.models.deepseek_v2 import patch_deepseek_config_json_for_tokenizer

        patch_deepseek_config_json_for_tokenizer(tokenizer_path)
    except Exception:
        pass
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    text = _apply_chat(tok, args.prompt, args.chat_template)
    input_ids = _encode(tok, text)
    if input_ids.shape[1] == 0:
        print("[Error] empty input_ids")
        return 2
    input_ids_cpu = input_ids.cpu()

    print(f"[Info] tokens={input_ids.shape[1]} chat_template={args.chat_template}")
    _print_path_info(args.lite_model, hf_model, tokenizer_path)

    logits_hf: torch.Tensor | None = None
    if args.load_hf_hiddens:
        print(f"[Step] Loading HF data from {args.load_hf_hiddens}")
        blob = _torch_load(args.load_hf_hiddens)
        if isinstance(blob, dict) and "checkpoints" in blob:
            hf_h = blob["checkpoints"]
            logits_hf = blob.get("logits_last")
        else:
            hf_h = blob
    else:
        print("[Step 1/2] HF forward + hooks + logits (last token)...")
        hf_h, logits_hf = _run_hf_capture(cmp, hf_model, input_ids_cpu, args.dtype)
        if args.save_hf_hiddens:
            torch.save({"checkpoints": hf_h, "logits_last": logits_hf}, args.save_hf_hiddens)
            print(f"[Info] saved HF checkpoints + logits to {args.save_hf_hiddens}")

    if args.skip_lite:
        print("[Info] --skip-lite: done.")
        return 0

    n_layers = len([k for k in hf_h if k.startswith("layer_")])
    keys = _ordered_checkpoint_keys(n_layers)

    if not args.load_hf_hiddens:
        gc.collect()
        torch.cuda.empty_cache()

    print("[Step 2/2] Lite (GGUF) forward + hooks + logits (last token)...")
    lite_h, logits_lite = _run_lite_capture(cmp, args.lite_model, input_ids_cpu, args.max_model_len)

    if set(hf_h.keys()) != set(lite_h.keys()):
        print(f"[Error] key mismatch HF={sorted(hf_h.keys())} Lite={sorted(lite_h.keys())}")
        return 2

    print("\n=== Per-checkpoint: last-token hidden (HF vs Lite) ===")
    print(f"{'checkpoint':<14} {'cosine':>10} {'mse':>12} {'mae':>12} {'d_cos':>10}")
    first_warn: str | None = None
    prev_cos: float | None = None
    for k in keys:
        cos, mse, mae = _hidden_metrics(hf_h[k], lite_h[k])
        d_cos = ""
        if prev_cos is not None:
            d_cos = f"{cos - prev_cos:+.6f}"
        prev_cos = cos
        print(f"{k:<14} {cos:10.6f} {mse:12.6e} {mae:12.6e} {d_cos:>10}")
        if first_warn is None and cos < args.cosine_warn:
            first_warn = k

    if first_warn:
        print(
            f"\n[Readout] First checkpoint with cosine < {args.cosine_warn}: {first_warn} "
            "(inspect mapping vs quantization; embed-first → tokenizer/embed; mid-layer → block math.)"
        )
    else:
        print(f"\n[Readout] All checkpoints cosine >= {args.cosine_warn}.")

    if logits_hf is None:
        print("\n[Info] HF logits not in loaded file; running HF once for logits comparison only...")
        logits_hf = cmp.run_hf(hf_model, tokenizer_path, input_ids_cpu, args.dtype)

    print("\n=== Last-token logits (HF vs Lite) ===")
    stats = _topk_diff(logits_hf, logits_lite, k=32)
    for kk, vv in stats.items():
        print(f"  {kk}: {vv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
