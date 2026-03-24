#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Small-sample logits / greedy-token comparison: HuggingFace (bf16/fp16) vs FastInference Lite (GGUF).

Use the same tokenizer and the same token id sequence for both runs (sequential on GPU by default
to reduce VRAM: unload HF before loading Lite).

Example:
  uv run python tests/tools/compare_hf_lite_deepseek_logits.py \\
    --lite-model models/DeepSeek-V2-Lite-GGUF \\
    --hf-model models/DeepSeek-V2-Lite-Chat \\
    --chat-template auto \\
    --check-mla-decode \\
    --prompt "What is the capital of France? Answer in a few words."

HF loads via ``transformers.models.deepseek_v2.DeepseekV2ForCausalLM`` when available (no remote code).
Use ``--check-mla-decode`` to verify MLA incremental decode matches a single full forward on the same GGUF weights.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _default_hf_model() -> str:
    """Prefer local `models/DeepSeek-V2-Lite-Chat` when present (same layout as sibling GGUF)."""
    local = os.path.join(REPO_ROOT, "models", "DeepSeek-V2-Lite-Chat")
    if os.path.isdir(local) and os.path.isfile(os.path.join(local, "config.json")):
        return local
    return "deepseek-ai/DeepSeek-V2-Lite-Chat"


def _resolved_tokenizer_path(lite_model: str, explicit_tokenizer: str | None) -> str:
    """Match engine behavior: sibling HF Chat dir when not overridden."""
    base = explicit_tokenizer or lite_model
    if explicit_tokenizer is not None:
        return base
    try:
        from vllm.model_executor.models.deepseek_hf_reference import (
            resolve_deepseek_hf_chat_dir,
        )

        ref = resolve_deepseek_hf_chat_dir(lite_model)
        if ref:
            return ref
    except Exception:
        pass
    return base


def _apply_chat_template_if_needed(tokenizer, prompt: str, mode: str) -> str:
    if mode == "off":
        return prompt
    tpl = getattr(tokenizer, "chat_template", None)
    if not tpl:
        return prompt
    try:
        if mode == "on":
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        # auto
        s = prompt.lstrip()
        if len(s) >= 12 and "<|im_start|>" in s[:400]:
            return prompt
        if s.startswith("<|") and "user" in s[:120].lower():
            return prompt
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt


def _encode(tokenizer, text: str) -> torch.Tensor:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids and getattr(tokenizer, "bos_token_id", None) is not None:
        ids = tokenizer.encode(text, add_special_tokens=True)
    return torch.tensor([ids], dtype=torch.long, device="cuda")


def _regression_gate_result(gate: str, stats: dict) -> tuple[int, str]:
    """Non-zero exit for CI/regression when gate is set. See docs/INFERENCE_ACCURACY.md (DeepSeek GGUF vs HF)."""
    if gate == "none":
        return 0, ""
    if gate == "safetensors":
        ok = stats["cosine"] >= 0.998 and bool(stats["argmax_match"])
        msg = (
            f"\n[Regression-gate safetensors] {'PASS' if ok else 'FAIL'} "
            f"(need CosSim>=0.998 and argmax match; got cosine={stats['cosine']:.6f})."
        )
        return (0 if ok else 1), msg
    if gate == "deepseek-gguf":
        # Q4 GGUF vs bf16 reference: logits drift; treat argmax / cosine / top-k overlap as composite health.
        min_cos = float(os.environ.get("FASTINFERENCE_DEEPSEEK_GGUF_REGRESSION_MIN_COS", "0.30"))
        min_topk = int(os.environ.get("FASTINFERENCE_DEEPSEEK_GGUF_REGRESSION_MIN_TOPK", "20"))
        ok = (
            bool(stats["argmax_match"])
            or float(stats["cosine"]) >= min_cos
            or int(stats["topk_idx_overlap"]) >= min_topk
        )
        msg = (
            f"\n[Regression-gate deepseek-gguf] {'PASS' if ok else 'FAIL'} "
            f"(argmax_match OR cosine>={min_cos} OR topk_overlap>={min_topk}; "
            f"got cosine={stats['cosine']:.6f}, topk={stats['topk_idx_overlap']}, argmax_match={stats['argmax_match']}). "
            f"Same-safetensors parity is stricter — use --regression-gate safetensors on DeepSeek-V2-Lite-Chat."
        )
        return (0 if ok else 1), msg
    return 0, ""


def _topk_diff(a: torch.Tensor, b: torch.Tensor, k: int) -> dict:
    """a, b: 1d same length."""
    diff = (a.float() - b.float()).abs()
    mse = float((diff ** 2).mean().item())
    # cosine on full vocab (avoid zero vec)
    na = a.float().norm()
    nb = b.float().norm()
    cos = float((a.float() @ b.float()) / (na * nb + 1e-8)) if na > 0 and nb > 0 else 0.0
    ta = torch.topk(a, k=min(k, a.numel()))
    tb = torch.topk(b, k=min(k, b.numel()))
    sa, ia = ta.values.tolist(), ta.indices.tolist()
    sb, ib = tb.values.tolist(), tb.indices.tolist()
    overlap = len(set(ia) & set(ib))
    return {
        "mse": mse,
        "mae": float(diff.max().item()),
        "cosine": cos,
        "topk_idx_overlap": overlap,
        "argmax_a": int(a.argmax().item()),
        "argmax_b": int(b.argmax().item()),
        "argmax_match": int(a.argmax().item()) == int(b.argmax().item()),
    }


def _build_lite_attn_metadata(
    seq_len: int,
    num_layers: int,
    max_model_len: int,
    device: torch.device,
) -> dict:
    block_size = 16
    num_blocks_per_seq = max(1, max_model_len // block_size)
    block_table = torch.arange(0, num_blocks_per_seq, device=device, dtype=torch.int32)
    slot_mapping = torch.arange(seq_len, device=device, dtype=torch.long)
    return {
        "slot_mapping": slot_mapping,
        "seq_lens": torch.tensor([seq_len], device=device, dtype=torch.int32),
        "is_prefill": True,
        "kv_start_indices": torch.tensor([0], device=device, dtype=torch.int32),
        "block_tables": block_table.unsqueeze(0),
        "linear_attn_carry": [None] * num_layers,
        "linear_conv_carry": [None] * num_layers,
    }


def _lite_forward_logits_last(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    num_layers: int,
    max_model_len: int,
) -> torch.Tensor:
    device = input_ids.device
    seq = input_ids.shape[1]
    positions = torch.arange(seq, device=device, dtype=torch.long).unsqueeze(0)
    kv_caches = [None] * num_layers
    meta = _build_lite_attn_metadata(seq, num_layers, max_model_len, device)
    with torch.inference_mode():
        logits = model(input_ids, positions, kv_caches, meta, lora_mapping=None)
    return logits[0, -1, :].float().cpu()


def _load_hf_causal_lm(hf_model: str, dtype: str):
    """Prefer stock Transformers DeepSeek V2 (avoids remote-code / transformers version skew)."""
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    try:
        from transformers.models.deepseek_v2 import DeepseekV2ForCausalLM

        return DeepseekV2ForCausalLM.from_pretrained(
            hf_model,
            torch_dtype=torch_dtype,
            device_map="cuda",
            attn_implementation="eager",
        )
    except Exception:
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            hf_model,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="cuda",
            attn_implementation="eager",
        )


def _alloc_mla_kv(
    model: torch.nn.Module,
    max_model_len: int,
    n_layers: int,
    device: torch.device,
) -> list:
    hf = model.model.config
    nh = int(getattr(hf, "num_attention_heads", 16))
    qk = int(getattr(hf, "qk_nope_head_dim", 128)) + int(getattr(hf, "qk_rope_head_dim", 64))
    vd = int(getattr(hf, "v_head_dim", 128))
    act_dtype = next(model.parameters()).dtype
    mla_kv = []
    for _ in range(n_layers):
        mla_kv.append(
            (
                torch.zeros(max_model_len, nh, qk, device=device, dtype=act_dtype),
                torch.zeros(max_model_len, nh, vd, device=device, dtype=act_dtype),
            )
        )
    return mla_kv


def _lite_logits_full_vs_mla_decode(
    model: torch.nn.Module,
    prompt_ids_1d: torch.Tensor,
    first_token_id: int,
    num_layers: int,
    max_model_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compare last-position logits for sequence [prompt, first_token]:
    (A) one forward with kv_caches=None (full prefill attention)
    (B) prompt prefill into mla_kv + single-token decode (LiteEngine path)
    Same weights -> (A) and (B) should match; if only (B) diverges from HF, MLA decode is suspect.
    """
    device = prompt_ids_1d.device
    T = int(prompt_ids_1d.shape[0])
    prompt = prompt_ids_1d.unsqueeze(0)
    kv_none = [None] * num_layers
    block_size = 16
    num_blocks = max(1, max_model_len // block_size)
    block_table = torch.arange(0, num_blocks, device=device, dtype=torch.int32).unsqueeze(0)

    full = torch.cat(
        [prompt, torch.tensor([[first_token_id]], device=device, dtype=torch.long)],
        dim=1,
    )
    pos_full = torch.arange(T + 1, device=device, dtype=torch.long).unsqueeze(0)
    meta_full = {
        "slot_mapping": torch.arange(T + 1, device=device, dtype=torch.long),
        "seq_lens": torch.tensor([T + 1], device=device, dtype=torch.int32),
        "is_prefill": True,
        "kv_start_indices": torch.tensor([0], device=device, dtype=torch.int32),
        "block_tables": block_table,
        "linear_attn_carry": [None] * num_layers,
        "linear_conv_carry": [None] * num_layers,
    }
    with torch.inference_mode():
        logits_full = model(full, pos_full, kv_none, meta_full, lora_mapping=None)[0, -1, :].float().cpu()

    mla_kv = _alloc_mla_kv(model, max_model_len, num_layers, device)
    pos_p = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0)
    meta_p = {
        "slot_mapping": torch.arange(T, device=device, dtype=torch.long),
        "seq_lens": torch.tensor([T], device=device, dtype=torch.int32),
        "is_prefill": True,
        "kv_start_indices": torch.tensor([0], device=device, dtype=torch.int32),
        "block_tables": block_table,
        "linear_attn_carry": [None] * num_layers,
        "linear_conv_carry": [None] * num_layers,
        "mla_kv": mla_kv,
        "mla_prefill_kv_range": (0, T),
    }
    pos_d = torch.tensor([[T]], device=device, dtype=torch.long)
    meta_d = {
        "slot_mapping": torch.tensor([T], device=device, dtype=torch.long),
        "seq_lens": torch.tensor([T + 1], device=device, dtype=torch.int32),
        "is_prefill": False,
        "kv_start_indices": torch.tensor([T], device=device, dtype=torch.int32),
        "block_tables": block_table,
        "linear_attn_carry": [None] * num_layers,
        "linear_conv_carry": [None] * num_layers,
        "mla_kv": mla_kv,
        "mla_cached_len": T,
    }
    one = torch.tensor([[first_token_id]], device=device, dtype=torch.long)
    with torch.inference_mode():
        _ = model(prompt, pos_p, kv_none, meta_p, lora_mapping=None)
        logits_mla = model(one, pos_d, kv_none, meta_d, lora_mapping=None)[0, -1, :].float().cpu()
    return logits_full, logits_mla


def run_hf(
    hf_model: str,
    _tokenizer_path: str,
    input_ids_cpu: torch.Tensor,
    dtype: str,
) -> torch.Tensor:
    # _tokenizer_path: kept for call-site compatibility; HF loads from hf_model.
    model = _load_hf_causal_lm(hf_model, dtype)
    model.eval()
    ids = input_ids_cpu.to(device="cuda", dtype=torch.long)
    with torch.inference_mode():
        out = model(input_ids=ids)
        logits_last = out.logits[0, -1, :].float().cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return logits_last


def load_lite_model(lite_model: str, max_model_len: int):
    from vllm.config import CacheConfig, LoadConfig, ModelConfig, SchedulerConfig, VllmConfig
    from vllm.model_executor.layers.quantization.gguf import GGUFConfig
    from vllm.model_executor.model_loader import get_model

    m_cfg = ModelConfig(model=lite_model, tokenizer=lite_model)
    c_cfg = CacheConfig(block_size=16, gpu_memory_utilization=0.5, swap_space=4)
    s_cfg = SchedulerConfig(
        max_num_batched_tokens=8192,
        max_num_seqs=4,
        max_model_len=max_model_len,
    )
    l_cfg = LoadConfig()
    q_cfg = GGUFConfig()
    v_cfg = VllmConfig(m_cfg, c_cfg, s_cfg, l_cfg, quant_config=q_cfg)
    model = get_model(vllm_config=v_cfg)
    model.eval()
    return model


def run_lite(
    lite_model: str,
    input_ids_cpu: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    model = load_lite_model(lite_model, max_model_len)
    n_layers = int(getattr(model.model.config, "num_hidden_layers", 0))
    ids = input_ids_cpu.to(device="cuda", dtype=torch.long)
    logits_last = _lite_forward_logits_last(model, ids, n_layers, max_model_len)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return logits_last


def greedy_tokens_hf(hf_model: str, tokenizer_path: str, input_ids: torch.Tensor, steps: int, dtype: str):
    from transformers import AutoTokenizer

    model = _load_hf_causal_lm(hf_model, dtype)
    model.eval()
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    ids = input_ids.to("cuda")
    out_ids = []
    with torch.inference_mode():
        for _ in range(steps):
            logits = model(input_ids=ids).logits[0, -1]
            t = int(logits.argmax().item())
            out_ids.append(t)
            ids = torch.cat([ids, torch.tensor([[t]], device=ids.device)], dim=1)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out_ids, [tok.decode([t]) for t in out_ids]


def greedy_tokens_lite(
    lite_model: str,
    input_ids: torch.Tensor,
    steps: int,
    max_model_len: int,
    tokenizer_path: str | None = None,
):
    from vllm.config import CacheConfig, LoadConfig, ModelConfig, SchedulerConfig, VllmConfig
    from vllm.model_executor.layers.quantization.gguf import GGUFConfig
    from vllm.model_executor.model_loader import get_model

    m_cfg = ModelConfig(model=lite_model, tokenizer=lite_model)
    c_cfg = CacheConfig(block_size=16, gpu_memory_utilization=0.5, swap_space=4)
    s_cfg = SchedulerConfig(max_num_batched_tokens=8192, max_num_seqs=4, max_model_len=max_model_len)
    l_cfg = LoadConfig()
    v_cfg = VllmConfig(m_cfg, c_cfg, s_cfg, l_cfg, quant_config=GGUFConfig())
    model = get_model(vllm_config=v_cfg)
    model.eval()
    n_layers = int(getattr(model.model.config, "num_hidden_layers", 0))
    tok_path = tokenizer_path or lite_model
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    ids = input_ids.to("cuda")
    out_ids = []
    with torch.inference_mode():
        for _ in range(steps):
            seq = ids.shape[1]
            positions = torch.arange(seq, device=ids.device, dtype=torch.long).unsqueeze(0)
            kv_caches = [None] * n_layers
            meta = _build_lite_attn_metadata(seq, n_layers, max_model_len, ids.device)
            logits = model(ids, positions, kv_caches, meta, lora_mapping=None)
            t = int(logits[0, -1].argmax().item())
            out_ids.append(t)
            ids = torch.cat([ids, torch.tensor([[t]], device=ids.device)], dim=1)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out_ids, [tok.decode([t]) for t in out_ids]


def main() -> int:
    p = argparse.ArgumentParser(description="HF vs Lite logits comparison (DeepSeek V2 Lite)")
    p.add_argument(
        "--hf-model",
        type=str,
        default=None,
        help="HF weights (repo id or local dir). Default: local models/DeepSeek-V2-Lite-Chat if present, else deepseek-ai/DeepSeek-V2-Lite-Chat",
    )
    p.add_argument(
        "--lite-model",
        type=str,
        required=True,
        help="Local GGUF directory (config.json + .gguf + tokenizer)",
    )
    p.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer path (default: same as --lite-model)",
    )
    p.add_argument("--prompt", type=str, default="What is 2+2? Answer with one digit.")
    p.add_argument(
        "--chat-template",
        choices=("off", "auto", "on"),
        default="off",
        help="Wrap prompt as chat user turn (same idea as quality_bar_spotcheck)",
    )
    p.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--topk", type=int, default=32)
    p.add_argument(
        "--greedy-steps",
        type=int,
        default=4,
        help="Greedy decode steps to compare token-by-token (0 = skip)",
    )
    p.add_argument(
        "--skip-hf",
        action="store_true",
        help="Only run Lite (for debugging)",
    )
    p.add_argument(
        "--skip-lite",
        action="store_true",
        help="Only run HF (for debugging)",
    )
    p.add_argument(
        "--check-mla-decode",
        action="store_true",
        help=(
            "Reload Lite once: compare logits for [prompt,t0] via (1) single full forward vs "
            "(2) MLA prefill + one decode step; t0 = HF greedy argmax on prompt. "
            "If (1)≈(2) but HF≉Lite, suspect GGUF/RoPE/weights; if (1)≉(2), fix MLA decode."
        ),
    )
    p.add_argument(
        "--regression-gate",
        choices=("none", "safetensors", "deepseek-gguf"),
        default="none",
        help=(
            "Exit non-zero on mismatch: "
            "safetensors=CosSim>=0.998 and argmax match (bf16 implementation parity); "
            "deepseek-gguf=Q4 GGUF vs bf16 HF (argmax OR cosine>=env MIN OR topk overlap; see env vars in script)."
        ),
    )
    args = p.parse_args()
    if args.skip_hf and args.skip_lite:
        print("[Error] Cannot use --skip-hf and --skip-lite together.")
        return 2

    if args.hf_model is None:
        args.hf_model = _default_hf_model()

    tokenizer_path = _resolved_tokenizer_path(args.lite_model, args.tokenizer)
    try:
        from vllm.model_executor.models.deepseek_v2 import patch_deepseek_config_json_for_tokenizer

        patch_deepseek_config_json_for_tokenizer(tokenizer_path)
    except Exception:
        pass
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    text = _apply_chat_template_if_needed(tok, args.prompt, args.chat_template)
    input_ids = _encode(tok, text)
    if input_ids.shape[1] == 0:
        print("[Error] Empty input_ids after encode.")
        return 2
    input_ids_cpu = input_ids.cpu()
    print(f"[Info] prompt chars={len(text)} tokens={input_ids.shape[1]} chat_template={args.chat_template}")
    print(f"[Info] tokenizer={tokenizer_path}")

    if not args.skip_hf and not args.skip_lite:
        print("[Step 1/2] HF forward (last-position logits)...")
        logits_hf = run_hf(args.hf_model, tokenizer_path, input_ids_cpu, args.dtype)
        print("[Step 2/2] Lite forward (last-position logits)...")
        logits_lite = run_lite(args.lite_model, input_ids_cpu, args.max_model_len)
        stats = _topk_diff(logits_hf, logits_lite, k=args.topk)
        print("\n=== Last-token logits (HF vs Lite) ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        if stats["mse"] < 1e-4 and stats["argmax_match"]:
            print("\n[Readout] Very close — differences likely quantization/numerics.")
        elif stats["cosine"] > 0.99 and stats["argmax_match"]:
            print("\n[Readout] Strong match — Q4 drift may still change longer greedy paths.")
        elif stats["cosine"] < 0.95 or not stats["argmax_match"]:
            print("\n[Readout] Large gap — inspect GGUF mapping / Lite path before blaming sampling.")

        rc_gate, gate_msg = _regression_gate_result(args.regression_gate, stats)
        if gate_msg:
            print(gate_msg)
        if rc_gate != 0:
            return rc_gate

        if args.check_mla_decode:
            first_t = int(logits_hf.argmax().item())
            print(
                f"\n=== Lite self-check: MLA decode vs full prefill (t0={first_t}, HF argmax on prompt) ==="
            )
            m = load_lite_model(args.lite_model, args.max_model_len)
            try:
                n_layers = int(getattr(m.model.config, "num_hidden_layers", 0))
                prompt_1d = input_ids_cpu[0].to(device="cuda", dtype=torch.long)
                lf, lm = _lite_logits_full_vs_mla_decode(
                    m, prompt_1d, first_t, n_layers, args.max_model_len
                )
                st_mla = _topk_diff(lf, lm, k=args.topk)
                for k, v in st_mla.items():
                    print(f"  {k}: {v}")
                if st_mla.get("cosine", 0.0) > 0.999 and st_mla.get("argmax_match"):
                    print(
                        "\n[Readout] MLA decode matches full prefill on Lite — if HF vs Lite differs, "
                        "focus on GGUF mapping / quantization vs HF bf16."
                    )
                else:
                    print(
                        "\n[Readout] MLA decode diverges from full prefill — fix MLA incremental path "
                        "(LiteEngine decode) before interpreting HF vs Lite gap."
                    )
            finally:
                del m
                gc.collect()
                torch.cuda.empty_cache()

    elif args.skip_hf:
        logits_lite = run_lite(args.lite_model, input_ids_cpu, args.max_model_len)
        print("Lite logits last (first 8):", logits_lite[:8].tolist())
        return 0
    else:
        logits_hf = run_hf(args.hf_model, tokenizer_path, input_ids_cpu, args.dtype)
        print("HF logits last (first 8):", logits_hf[:8].tolist())
        return 0

    if args.greedy_steps > 0 and not args.skip_hf and not args.skip_lite:
        print(f"\n=== Greedy next {args.greedy_steps} tokens (HF then Lite, reload each) ===")
        ids_hf, pieces_hf = greedy_tokens_hf(
            args.hf_model, tokenizer_path, input_ids_cpu, args.greedy_steps, args.dtype
        )
        ids_lite, pieces_lite = greedy_tokens_lite(
            args.lite_model,
            input_ids_cpu,
            args.greedy_steps,
            args.max_model_len,
            tokenizer_path=tokenizer_path,
        )
        match = ids_hf == ids_lite
        print(f"  HF   ids: {ids_hf}")
        print(f"  Lite ids: {ids_lite}")
        print(f"  all_match: {match}")
        print(f"  HF   pieces: {pieces_hf}")
        print(f"  Lite pieces: {pieces_lite}")
        if args.regression_gate == "safetensors" and not match:
            print("\n[Regression-gate safetensors] FAIL (greedy token mismatch).")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
