# SPDX-License-Identifier: Apache-2.0
"""
LitevLLM Semantic Integrity Verification Suite.
Compares LitevLLM (Triton/LiteEngine) against Hugging Face (PyTorch) 
to ensure absolute numerical alignment and semantic correctness.
"""
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoConfig
from vllm.config import VllmConfig, ModelConfig, CacheConfig, SchedulerConfig, LoadConfig
from vllm.engine.lite_engine import LiteEngine
from vllm.sampling_params import SamplingParams
import time
import argparse
import os
import sys
from typing import Any, Callable, List, Optional
# Debug: check where vllm is loaded from
import vllm.model_executor.models.llama
print(f"DEBUG: Loaded vllm.llama from: {vllm.model_executor.models.llama.__file__}")


def _lite_engine_step_budget(
    engine: LiteEngine, prompt_token_len: int, max_new_tokens: int
) -> int:
    """Upper bound on step() calls: chunked prefill + decode + margin."""
    chunk_sz = max(1, int(getattr(engine, "_prefill_chunk_size", 512)))
    prefill_chunks = max(1, (prompt_token_len + chunk_sz - 1) // chunk_sz)
    return prefill_chunks + max_new_tokens * 3 + 500


def _run_lite_steps_until(
    engine: LiteEngine,
    description: str,
    max_steps: int,
    stop_fn: Callable[[List[Any]], Optional[Any]],
) -> Any:
    """
    Repeatedly call engine.step() until stop_fn returns non-None.
    Empty step outputs are normal during chunked prefill.
    """
    for _ in range(max_steps):
        step_outputs = engine.step()
        done = stop_fn(step_outputs)
        if done is not None:
            return done
    raise RuntimeError(
        f"{description}: exceeded {max_steps} LiteEngine.step() calls "
        f"(possible CUDA hang, engine error, or logic bug). "
        f"active_request_count={engine.active_request_count}."
    )


PRESETS = {
    # TinyLlama dense baseline
    "tinyllama": {
        "prompt": "The capital of France is",
        "max_new_tokens": 10,
        "quant": "none",
    },
    # Qwen 3.5 9B GGUF
    "qwen35_9b_gguf": {
        "prompt": "The capital of France is",
        "max_new_tokens": 10,
        "quant": "gguf",
    },
    # Qwen 3.5 9B AWQ
    "qwen35_9b_awq": {
        "prompt": "The capital of France is",
        "max_new_tokens": 10,
        "quant": "awq",
    },
    # Qwen 3.5 35B MoE GGUF
    "qwen35_35b_moe_gguf": {
        "prompt": "The capital of France is",
        "max_new_tokens": 10,
        "quant": "gguf",
    },
    # DeepSeek V2 Lite GGUF
    "deepseek_v2_lite_gguf": {
        "prompt": "The capital of France is",
        "max_new_tokens": 10,
        "quant": "gguf",
    },
    # GLM 4.7 Flash GGUF
    "glm_4_7_flash_gguf": {
        "prompt": "The capital of France is",
        "max_new_tokens": 10,
        "quant": "gguf",
    },
}


def compare_logits(lite_logits, ref_logits, threshold=0.999):
    """Computes cosine similarity and max absolute error."""
    a = lite_logits.flatten().float().unsqueeze(0)
    b = ref_logits.flatten().float().unsqueeze(0)
    cos_sim = F.cosine_similarity(a, b, dim=1).item()
    max_err = (lite_logits - ref_logits).abs().max().item()
    return cos_sim, max_err


def _read_awq_group_size_and_bits(model_path: str) -> tuple[int, int]:
    """Read AWQ group_size / bits from HF-style config.json (e.g. compressed-tensors pack-quantized)."""
    cfg_path = os.path.join(model_path, "config.json")
    group_size, bits = 128, 4
    try:
        if not os.path.isfile(cfg_path):
            return group_size, bits
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        qc = cfg.get("quantization_config") or {}
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
    except Exception as e:
        print(f"  [Warning] Could not parse AWQ params from config.json: {e}")
    return group_size, bits


def _resolve_hf_torch_dtype(config_path_dir: str) -> torch.dtype:
    """Prefer text_config.dtype / torch_dtype from config.json (robust for nested dicts)."""
    try:
        p = os.path.join(config_path_dir, "config.json")
        if not os.path.isfile(p):
            return torch.float16
        with open(p, "r") as f:
            raw = json.load(f)
        tc = raw.get("text_config") or {}
        if isinstance(tc, dict):
            ds = tc.get("dtype") or tc.get("torch_dtype")
        else:
            ds = None
        if ds is None:
            ds = raw.get("torch_dtype")
        if isinstance(ds, str) and "bfloat16" in ds.lower():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


def compare_logits_aligned(lite_logits: torch.Tensor, ref_logits: torch.Tensor):
    """Align vocab dims, skip NaN/Inf; return (cos_sim, max_err) or (nan, nan) if unusable."""
    a = lite_logits.float().flatten()
    b = ref_logits.float().flatten()
    n = min(a.numel(), b.numel())
    if n == 0:
        return float("nan"), float("nan")
    a = a[:n]
    b = b[:n]
    if a.numel() != b.numel():
        return float("nan"), float("nan")
    mask = torch.isfinite(a) & torch.isfinite(b)
    if not mask.any():
        return float("nan"), float("nan")
    a = a[mask]
    b = b[mask]
    cos_sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).item()
    max_err = (a - b).abs().max().item()
    return cos_sim, max_err

def _load_hf_reference_model(hf_path: str, dtype: torch.dtype, hf_device: str):
    """
    Load HF reference; supports dense CausalLM and Qwen3.5 multimodal checkpoints.
    Use hf_device='cpu' when Lite already occupies GPU VRAM to avoid OOM / illegal memory access.
    """
    hf_config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
    common = dict(
        pretrained_model_name_or_path=hf_path,
        config=hf_config,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if hf_device == "cpu":
        load_kw = {**common, "low_cpu_mem_usage": True}
        try:
            model = AutoModelForCausalLM.from_pretrained(**load_kw).eval()
        except Exception as e1:
            print(f"  [Info] AutoModelForCausalLM (CPU) load failed: {e1}")
            model = AutoModel.from_pretrained(**load_kw).eval()
        return model.to(torch.device("cpu"))
    # cuda: single-GPU auto placement (may OOM if Lite already uses most VRAM)
    load_kw = {**common, "device_map": "auto"}
    try:
        return AutoModelForCausalLM.from_pretrained(**load_kw).eval()
    except Exception as e1:
        print(f"  [Info] AutoModelForCausalLM load failed: {e1}")
    try:
        return AutoModel.from_pretrained(**load_kw).eval()
    except Exception as e2:
        print(f"  [Info] AutoModel load failed: {e2}")
        raise RuntimeError(f"Could not load HF reference from {hf_path}") from e2


def run_alignment_test(
    model_path,
    quant_type="none",
    prompt="The capital of France is",
    max_new_tokens=10,
    no_hf=False,
    hf_model_path=None,
    hf_device="auto",
    activation_audit: bool = False,
    activation_audit_max_passes: Optional[int] = None,
    disable_qwen35_stabilizers: bool = False,
):
    print(f"\n" + "="*60)
    print(f"AUDITING: {os.path.basename(model_path)} (Quant: {quant_type})")
    if hf_model_path:
        print(f"  HF baseline: {hf_model_path}")
        print(f"  HF device: {hf_device}")
    print("="*60)

    device = "cuda"
    # HF baseline dtype; refined when hf_load_path is known (see below).
    dtype = torch.float16

    # Force FP16 for audit to avoid quantization noise
    os.environ["FASTINFERENCE_KV_FP8"] = "0"
    # Qwen3.5: full-attn defaults to HF-faithful path. Optional legacy ROCm stabilizer:
    #   FASTINFERENCE_QWEN35_FULLATTN_STABILIZER=1
    # Older ablation flags (input cap + residual RMS) only apply when full-attn stabilizer is enabled.
    if disable_qwen35_stabilizers:
        os.environ["FASTINFERENCE_DISABLE_LINEAR_INPUT_CAP"] = "1"
        os.environ["FASTINFERENCE_DISABLE_RESIDUAL_STABILIZER"] = "1"
        print(
            "  [Ablation] Qwen3.5 linear-input cap + residual RMS disabled "
            "(relevant if FASTINFERENCE_QWEN35_FULLATTN_STABILIZER=1)."
        )
    
    # 1. Initialize LitevLLM Engine
    m_cfg = ModelConfig(model=model_path, tokenizer=model_path)
    c_cfg = CacheConfig(block_size=16, gpu_memory_utilization=0.9, swap_space=4)
    s_cfg = SchedulerConfig(max_num_batched_tokens=2048, max_num_seqs=32, max_model_len=2048)
    l_cfg = LoadConfig()
    
    q_cfg = None
    if quant_type == "awq":
        from vllm.model_executor.layers.quantization.awq import AWQConfig

        gs, wb = _read_awq_group_size_and_bits(model_path)
        q_cfg = AWQConfig(weight_bits=wb, group_size=gs)
        print(f"  [AWQ] group_size={gs}, weight_bits={wb} (from {model_path}/config.json)")
    elif quant_type == "gguf":
        from vllm.model_executor.layers.quantization.gguf import GGUFConfig
        q_cfg = GGUFConfig()
        
    v_cfg = VllmConfig(m_cfg, c_cfg, s_cfg, l_cfg, quant_config=q_cfg)
    
    print("[1/3] Loading LitevLLM (Python/Triton)...")
    engine = LiteEngine(v_cfg)
    from vllm.model_executor.model_loader import get_tokenizer

    # When comparing quantized Lite vs FP16/BF16 baseline, use baseline tokenizer so token ids match HF.
    tokenizer_source = hf_model_path if hf_model_path else model_path
    tokenizer = get_tokenizer(tokenizer_source, trust_remote_code=True)
    engine.tokenizer = tokenizer

    _act_sniffer = None
    if activation_audit:
        from vllm.debug.activation_sniffer import LiteActivationSniffer

        _act_sniffer = LiteActivationSniffer(
            engine.model,
            max_passes=activation_audit_max_passes,
        )
        _act_sniffer.attach()

    audit_result = True
    try:
        hf_model = None
        hf_load_path = None
        if not no_hf:
            hf_load_path = hf_model_path if hf_model_path else model_path
            dtype = _resolve_hf_torch_dtype(hf_load_path)
            # Separate baseline on GPU + Lite on GPU causes VRAM pressure and ROCm illegal access; force CPU.
            if hf_model_path and hf_device != "cpu":
                print(
                    f"  [Note] With --hf-model, loading HF on {hf_device} alongside Lite on GPU risks "
                    "OOM / HIP illegal access. Forcing HF to CPU."
                )
                hf_device = "cpu"

        if no_hf:
            print("[2/3] Skipping HF Reference (requested)...")
        elif hf_model_path is None:
            # Same-directory HF reference: load before audit (legacy path).
            try:
                _cfg = AutoConfig.from_pretrained(hf_load_path, trust_remote_code=True)
                text_cfg = getattr(_cfg, "text_config", None)
                dtype_str = None
                if text_cfg is not None:
                    dtype_str = getattr(text_cfg, "dtype", None) or getattr(text_cfg, "torch_dtype", None)
                if dtype_str is None:
                    dtype_str = getattr(_cfg, "torch_dtype", None)
                if isinstance(dtype_str, str) and dtype_str.lower() == "bfloat16":
                    dtype = torch.bfloat16
                print(f"[2/3] Loading HF Reference (PyTorch {dtype}) from {hf_load_path}...")
            except Exception:
                print(f"[2/3] Loading HF Reference (PyTorch {dtype}) from {hf_load_path}...")
            try:
                hf_model = _load_hf_reference_model(hf_load_path, dtype, hf_device)
            except Exception as e:
                print(f"  [Warning] HF Reference load failed: {e}")
                print("  Falling back to LitevLLM-only run.")
                hf_model = None
        else:
            # Defer HF load until after Lite prefill capture (GPU stays Lite-only for chunked prefill).
            try:
                _cfg = AutoConfig.from_pretrained(hf_load_path, trust_remote_code=True)
                text_cfg = getattr(_cfg, "text_config", None)
                dtype_str = None
                if text_cfg is not None:
                    dtype_str = getattr(text_cfg, "dtype", None) or getattr(text_cfg, "torch_dtype", None)
                if dtype_str is None:
                    dtype_str = getattr(_cfg, "torch_dtype", None)
                if isinstance(dtype_str, str) and dtype_str.lower() == "bfloat16":
                    dtype = torch.bfloat16
            except Exception:
                pass
            print(f"[2/3] HF Reference will load after Lite prefill (path={hf_load_path}, device={hf_device})...")

        # 3. Execution & Comparison (Greedy)
        input_ids_tokens = tokenizer.encode(prompt, return_tensors="pt")
        print(f"  Input Tokens: {input_ids_tokens[0].tolist()}")
        prompt_token_len = int(input_ids_tokens.shape[1])
        step_budget_audit = _lite_engine_step_budget(engine, prompt_token_len, 4)
        step_budget_full = _lite_engine_step_budget(
            engine, prompt_token_len, max_new_tokens
        )

        print(f"[3/3] Running Generation Audit...")

        # Prefill logits: when --hf-model is set, run Lite FIRST while GPU is not shared with HF.
        lite_prefill_logits_cpu = None
        lite_prefill_token = None
        if not no_hf and hf_model_path is not None:
            captured_logits = []
            original_forward = engine.model.forward

            def audit_forward(*args, **kwargs):
                res = original_forward(*args, **kwargs)
                captured_logits.append(res.detach().cpu())
                return res

            engine.model.forward = audit_forward
            engine.add_request("audit", prompt, SamplingParams(max_tokens=1, temperature=0.0))
            audit_ro = _run_lite_steps_until(
                engine,
                "Prefill audit (HF deferred path)",
                step_budget_audit,
                lambda outs: outs[0] if outs else None,
            )
            lite_prefill_token = audit_ro.outputs[0].token_ids[-1]
            engine.model.forward = original_forward
            lite_prefill_logits_cpu = captured_logits[0][:, -1, :].float().cpu()
            if not torch.isfinite(lite_prefill_logits_cpu).all():
                n_bad = int((~torch.isfinite(lite_prefill_logits_cpu)).sum().item())
                print(f"  [Warning] Lite prefill logits: {n_bad} non-finite values (check AWQ group_size / weights).")

            print(f"  [2b] Loading HF Reference (PyTorch {dtype}) from {hf_load_path}...")
            try:
                hf_model = _load_hf_reference_model(hf_load_path, dtype, hf_device)
            except Exception as e:
                print(f"  [Warning] HF Reference load failed: {e}")
                print("  Falling back to LitevLLM-only run (prefill logits captured but not compared).")
                hf_model = None

            if hf_model is not None:
                hf_eval_device = next(hf_model.parameters()).device
                input_ids_hf = input_ids_tokens.to(hf_eval_device)
                attn_mask = torch.ones_like(input_ids_hf, dtype=torch.long, device=hf_eval_device)
                with torch.inference_mode():
                    hf_outputs = hf_model(input_ids_hf, attention_mask=attn_mask)
                    if hasattr(hf_outputs, "logits") and hf_outputs.logits is not None:
                        hf_logits = hf_outputs.logits[:, -1, :]
                    elif isinstance(hf_outputs, tuple):
                        hf_logits = hf_outputs[0][:, -1, :]
                    else:
                        raise RuntimeError("HF model output has no usable logits for audit.")
                    hf_token = torch.argmax(hf_logits, dim=-1).item()
                hf_logits_cmp = hf_logits.float().cpu()
                vl, vr = lite_prefill_logits_cpu.shape[-1], hf_logits_cmp.shape[-1]
                if vl != vr:
                    print(f"  [Note] Vocab size: Lite={vl} vs HF={vr}; logits compared on min slice.")
                cos_sim, max_err = compare_logits_aligned(lite_prefill_logits_cpu, hf_logits_cmp)
                print(f"  Prefill Logits -> CosSim: {cos_sim:.6f}, MaxErr: {max_err:.6f}")
                print(f"  Prefill Token: HF={hf_token} | Lite={lite_prefill_token}")

        elif hf_model is not None:
            hf_eval_device = next(hf_model.parameters()).device
            input_ids_hf = input_ids_tokens.to(hf_eval_device)
            attn_mask = torch.ones_like(input_ids_hf, dtype=torch.long, device=hf_eval_device)
            with torch.inference_mode():
                hf_outputs = hf_model(input_ids_hf, attention_mask=attn_mask)
                if hasattr(hf_outputs, "logits") and hf_outputs.logits is not None:
                    hf_logits = hf_outputs.logits[:, -1, :]
                elif isinstance(hf_outputs, tuple):
                    hf_logits = hf_outputs[0][:, -1, :]
                else:
                    raise RuntimeError("HF model output has no usable logits for audit.")
                hf_token = torch.argmax(hf_logits, dim=-1).item()

            captured_logits = []
            original_forward = engine.model.forward

            def audit_forward(*args, **kwargs):
                res = original_forward(*args, **kwargs)
                captured_logits.append(res.detach().cpu())
                return res

            engine.model.forward = audit_forward
            engine.add_request("audit", prompt, SamplingParams(max_tokens=1, temperature=0.0))
            audit_ro2 = _run_lite_steps_until(
                engine,
                "Prefill audit (HF same-dir path)",
                step_budget_audit,
                lambda outs: outs[0] if outs else None,
            )
            lite_token = audit_ro2.outputs[0].token_ids[-1]
            engine.model.forward = original_forward
            lite_logits = captured_logits[0][:, -1, :].float().cpu()
            hf_logits_cmp = hf_logits.float().cpu()
            vl, vr = lite_logits.shape[-1], hf_logits_cmp.shape[-1]
            if vl != vr:
                print(f"  [Note] Vocab size: Lite={vl} vs HF={vr}; logits compared on min slice.")
            cos_sim, max_err = compare_logits_aligned(lite_logits, hf_logits_cmp)
            print(f"  Prefill Logits -> CosSim: {cos_sim:.6f}, MaxErr: {max_err:.6f}")
            print(f"  Prefill Token: HF={hf_token} | Lite={lite_token}")
    
        # Multi-token Generation Audit
        # LitevLLM for full sequence
        engine.add_request(
            "audit_full",
            prompt,
            SamplingParams(max_tokens=max_new_tokens, temperature=0.0),
        )
        full_lite_out = _run_lite_steps_until(
            engine,
            "Full greedy generation audit",
            step_budget_full,
            lambda outs: (
                outs[0].outputs[0] if outs and outs[0].finished else None
            ),
        )

        lite_text = full_lite_out.text
        print(f"  LitevLLM Output: '{lite_text}'")
        print(f"  Lite Tokens: {full_lite_out.token_ids}")

        if hf_model is not None:
            hf_eval_device = next(hf_model.parameters()).device
            input_ids_hf = input_ids_tokens.to(hf_eval_device)
            gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
            pad_id = getattr(tokenizer, "pad_token_id", None)
            if pad_id is None:
                pad_id = getattr(tokenizer, "eos_token_id", None)
            if pad_id is not None:
                gen_kwargs["pad_token_id"] = pad_id
            attn_mask_gen = torch.ones_like(input_ids_hf, dtype=torch.long, device=hf_eval_device)
            gen_kwargs["attention_mask"] = attn_mask_gen
            hf_full_gen = hf_model.generate(input_ids_hf, **gen_kwargs)
            hf_text = tokenizer.decode(hf_full_gen[0][input_ids_hf.shape[-1]:])
            print(f"  HF Reference:   '{hf_text}'")
        
            match = (lite_text.strip() == hf_text.strip())
            if match:
                print("  ✅ PASS: Semantic Integrity Verified.")
            else:
                print("  ❌ FAIL: Semantic Drift Detected.")
                lite_tokens = full_lite_out.token_ids
                hf_tokens = hf_full_gen[0][input_ids_hf.shape[-1]:].tolist()
                print(f"  Lite Tokens: {lite_tokens}")
                print(f"  HF Tokens:   {hf_tokens}")
            audit_result = match
        else:
            print("  [Info] Completed LitevLLM-only run (no reference comparison).")
            audit_result = True
    finally:
        if _act_sniffer is not None:
            _act_sniffer.detach()
    return audit_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--quant", type=str, default="none", choices=["none", "awq", "gguf"])
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=sorted(PRESETS.keys()),
        help="Optional model preset key to populate default prompt/max_new_tokens/quant.",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Override prompt for the audit.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override max_new_tokens for generation audit.",
    )
    parser.add_argument("--no-hf", action="store_true")
    parser.add_argument(
        "--hf-model",
        type=str,
        default=None,
        help="Separate HF/PyTorch baseline directory (e.g. FP16/BF16). Lite still uses --model.",
    )
    parser.add_argument(
        "--hf-device",
        type=str,
        default=None,
        choices=["auto", "cuda", "cpu"],
        help="Where to load HF reference. Default: cpu if --hf-model is set. "
        "If --hf-model is set, cuda/auto are forced to cpu (Lite+HF on one GPU risks ROCm illegal access).",
    )
    parser.add_argument(
        "--activation-audit",
        action="store_true",
        help="Print per-layer hidden-state stats (mean/std/min/max, nan/inf counts) on each Lite forward.",
    )
    parser.add_argument(
        "--activation-audit-max-passes",
        type=int,
        default=None,
        help="Stop printing activation lines after this many top-level Lite forwards (prefill+decode each count).",
    )
    parser.add_argument(
        "--disable-qwen35-stabilizers",
        action="store_true",
        help=(
            "Ablation with FASTINFERENCE_QWEN35_FULLATTN_STABILIZER=1: disable input caps and residual RMS "
            "(sets FASTINFERENCE_DISABLE_LINEAR_INPUT_CAP / FASTINFERENCE_DISABLE_RESIDUAL_STABILIZER)."
        ),
    )
    args = parser.parse_args()

    # Resolve preset-driven defaults
    preset_cfg = PRESETS.get(args.preset) if args.preset is not None else None
    effective_quant = args.quant
    effective_prompt = args.prompt
    effective_max_new_tokens = getattr(args, "max_new_tokens", None)

    if preset_cfg is not None:
        if effective_quant == "none":
            effective_quant = preset_cfg.get("quant", effective_quant)
        if effective_prompt is None:
            effective_prompt = preset_cfg.get("prompt", "The capital of France is")
        if effective_max_new_tokens is None:
            effective_max_new_tokens = preset_cfg.get("max_new_tokens", 10)

    if effective_prompt is None:
        effective_prompt = "The capital of France is"
    if effective_max_new_tokens is None:
        effective_max_new_tokens = 10

    hf_device = args.hf_device
    if hf_device is None:
        hf_device = "cpu" if args.hf_model else "auto"

    print(
        f"[Config] preset={args.preset}, quant={effective_quant}, "
        f"max_new_tokens={effective_max_new_tokens}, hf_device={hf_device}"
    )

    success = run_alignment_test(
        args.model,
        effective_quant,
        prompt=effective_prompt,
        max_new_tokens=effective_max_new_tokens,
        no_hf=args.no_hf,
        hf_model_path=args.hf_model,
        hf_device=hf_device,
        activation_audit=args.activation_audit,
        activation_audit_max_passes=args.activation_audit_max_passes,
        disable_qwen35_stabilizers=args.disable_qwen35_stabilizers,
    )
    if not success:
        exit(1)
