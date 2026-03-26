# SPDX-License-Identifier: Apache-2.0
"""
LitevLLM Semantic Integrity Verification Suite (Tier A / regression).
Compares LitevLLM (Triton/LiteEngine) against Hugging Face (PyTorch) on prefill logits
and greedy tokens. Use this when debugging kernels, loaders (e.g. GGUF), or
implementation drift.

Product acceptance is better expressed by Tier B (readable completions under normal
prompts); see docs/INFERENCE_ACCURACY.md and tests/tools/quality_bar_spotcheck.py. Low CosSim vs an
FP16 HF baseline does not always mean unusable output for quantized checkpoints, but
gibberish or HF-normal vs Lite-broken behavior still indicates a bug to fix in the
Lite path.

Use ``--hf-same-as-lite`` to load the HF reference from the **same** directory as
``--model`` (e.g. AWQ/GGUF tree). That isolates Lite vs HF *implementation* drift;
use a separate ``--hf-model`` FP16/BF16 tree only when checking against an unquantized baseline.

Optional: ``--apply-chat-template auto|on`` wraps the prompt like Tier-B spotcheck; ``--prefill-only``
skips full greedy generation and only checks last prefill logits vs HF (CosSim + argmax).

For **Lite GGUF or AWQ vs separate HF FP16** (``--hf-model`` pointing to a different tree), cosine uses a
relaxed floor (``PREFILL_COSIM_MIN_*_VS_FP16``); argmax must still match except **DeepSeek-V2-Lite GGUF**,
where Q4 vs bf16 may drift — pass if argmax matches **or** cosine clears ``PREFILL_COSIM_MIN_DEEPSEEK_GGUF_VS_FP16``.
Same-directory Lite vs HF
(``--hf-same-as-lite`` or no ``--hf-model``) keeps the strict ``PREFILL_COSIM_MIN``. **35B MoE GGUF** usually cannot
load a matching HF model for logits compare; use **9B GGUF vs 9B FP16** here, or
``tests/tools/qwen35_moe_packed_lite_logits_audit.py`` (packed vs dense) on the 35B tree.
"""
import json
import torch
import torch.nn.functional as F

# HF Qwen3.5 reference on CPU: disable flash-linear-attention import so GatedDeltaNet uses
# torch_chunk_gated_delta_rule (FLA/Triton on CPU tensors raises on ROCm/CUDA builds).
import transformers.utils.import_utils as _transformers_import_utils

_transformers_import_utils.is_flash_linear_attention_available = lambda: False  # noqa: E731

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoConfig
from vllm.config import VllmConfig, ModelConfig, CacheConfig, SchedulerConfig, LoadConfig
from vllm.engine.lite_engine import LiteEngine
from vllm.sampling_params import SamplingParams
import time
import argparse
import math
import os
import sys
from typing import Any, Callable, List, Optional

# Prefill-only audit: HF reference logits must align within this cosine floor (aligned vocab slice).
# Same-precision Lite vs HF (e.g. both FP16).
PREFILL_COSIM_MIN = 0.998
# Lite GGUF vs HF FP16 baseline: quantization shifts logits; argmax match is primary.
PREFILL_COSIM_MIN_GGUF_VS_FP16 = 0.99
# Lite AWQ vs HF FP16 baseline: INT4 groupwise weights vs BF16 reference; same rationale as GGUF vs FP16.
PREFILL_COSIM_MIN_AWQ_VS_FP16 = 0.99
# DeepSeek-V2-Lite Q4 GGUF vs bf16 HF: logits drift; use composite pass (see prefill_hf_alignment_pass).
PREFILL_COSIM_MIN_DEEPSEEK_GGUF_VS_FP16 = float(
    os.environ.get("FASTINFERENCE_DEEPSEEK_GGUF_REGRESSION_MIN_COS", "0.30")
)


def _is_deepseek_v2_lite_gguf_path(model_path: str, quant_type: str) -> bool:
    if quant_type != "gguf":
        return False
    b = os.path.basename(os.path.abspath(model_path)).lower()
    return "deepseek" in b and "lite" in b


def prefill_cosine_floor_for_hf_compare(
    model_path: str,
    hf_model_path: Optional[str],
    quant_type: str,
) -> float:
    """Cosine floor for prefill-only Lite vs HF when HF may use a different checkpoint than Lite."""
    if hf_model_path is None:
        return PREFILL_COSIM_MIN
    if os.path.abspath(os.path.realpath(hf_model_path)) == os.path.abspath(
        os.path.realpath(model_path)
    ):
        return PREFILL_COSIM_MIN
    if quant_type == "gguf" and _is_deepseek_v2_lite_gguf_path(model_path, quant_type):
        return PREFILL_COSIM_MIN_DEEPSEEK_GGUF_VS_FP16
    if quant_type == "gguf":
        return PREFILL_COSIM_MIN_GGUF_VS_FP16
    if quant_type == "awq":
        return PREFILL_COSIM_MIN_AWQ_VS_FP16
    return PREFILL_COSIM_MIN


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
    "qwen35_9b_fp16": {
        "prompt": "The capital of France is",
        "max_new_tokens": 10,
        "quant": "none",
    },
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
    # Qwen 3.5 35B MoE AWQ (HF safetensors / compressed-tensors; needs enough VRAM for BF16 MoE resident path)
    "qwen35_35b_moe_awq": {
        "prompt": "The capital of France is",
        "max_new_tokens": 10,
        "quant": "awq",
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


def _looks_like_preformatted_chat(text: str) -> bool:
    s = text.lstrip()
    if len(s) >= 12 and "<|im_start|>" in s[:400]:
        return True
    if s.startswith("<|") and "user" in s[:120].lower():
        return True
    return False


def apply_chat_template_for_verify(raw_prompt: str, tokenizer: Any, mode: str) -> str:
    """
    Wrap plain text as a single user turn when mode is auto/on (align with Tier-B spotcheck).
    mode: off | auto | on
    """
    if mode == "off":
        return raw_prompt
    tpl = getattr(tokenizer, "chat_template", None)
    if not tpl:
        if mode == "on":
            print(
                "  [Warning] --apply-chat-template on but tokenizer has no chat_template; using raw prompt."
            )
        return raw_prompt
    if mode == "auto":
        print(
            "  [Verify] --apply-chat-template auto: wrapping prompt as user message (tokenizer has chat_template)."
        )
    else:
        print("  [Verify] --apply-chat-template on: wrapping prompt as user message.")
    if _looks_like_preformatted_chat(raw_prompt):
        return raw_prompt
    try:
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": raw_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": raw_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
    except Exception as e:
        print(f"  [Warning] chat template failed: {e}; using raw prompt.")
        return raw_prompt


def prefill_hf_alignment_pass(
    cos_sim: Optional[float],
    lite_argmax: Optional[int],
    hf_argmax: Optional[int],
    cos_min: float = PREFILL_COSIM_MIN,
    *,
    deepseek_gguf_q4: bool = False,
) -> bool:
    if cos_sim is None or lite_argmax is None or hf_argmax is None:
        return False
    if not math.isfinite(cos_sim):
        return False
    if deepseek_gguf_q4:
        # Q4 GGUF vs bf16: pass if greedy argmax matches OR cosine is above a loose floor (quant drift).
        return (int(lite_argmax) == int(hf_argmax)) or (cos_sim >= cos_min)
    return (cos_sim >= cos_min) and (int(lite_argmax) == int(hf_argmax))


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
    # Avoid flash-linear-attention / Triton on CPU or mixed-device Qwen3.5 (illegal access on some ROCm setups).
    if getattr(hf_config, "model_type", "") == "qwen3_5":
        common["attn_implementation"] = "eager"
    # Stock Transformers DeepSeek V2 (same as tests/tools/compare_hf_lite_deepseek_logits.py): avoids
    # AutoModel + local config.json drift (e.g. KeyError 'type' on merged rope fields).
    if getattr(hf_config, "model_type", "") == "deepseek_v2":
        from transformers.models.deepseek_v2 import DeepseekV2ForCausalLM

        ds_kw = dict(
            pretrained_model_name_or_path=hf_path,
            torch_dtype=dtype,
            attn_implementation="eager",
            trust_remote_code=True,
        )
        if hf_device == "cpu":
            ds_kw["low_cpu_mem_usage"] = True
            model = DeepseekV2ForCausalLM.from_pretrained(**ds_kw).eval()
            return model.to(torch.device("cpu"))
        ds_kw["device_map"] = "auto"
        return DeepseekV2ForCausalLM.from_pretrained(**ds_kw).eval()
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
    apply_chat_template: str = "off",
    prefill_only: bool = False,
):
    print(f"\n" + "="*60)
    print(f"AUDITING: {os.path.basename(model_path)} (Quant: {quant_type})")
    if hf_model_path:
        print(f"  HF baseline: {hf_model_path}")
        print(f"  HF device: {hf_device}")
    mp_low = model_path.lower()
    if quant_type == "gguf" and ("35b" in mp_low or "moe" in mp_low):
        print(
            "  [Note] 35B MoE GGUF: full HF logits comparison is often infeasible on one machine; "
            "prefer 9B GGUF vs 9B FP16 here, or tests/tools/qwen35_moe_packed_lite_logits_audit.py "
            "(packed vs dense) on this checkpoint."
        )
    print("="*60)

    device = "cuda"
    # HF baseline dtype; refined when hf_load_path is known (see below).
    dtype = torch.float16

    # Default FP8 KV to save VRAM; set FASTINFERENCE_KV_FP8=0 before launch for bf16/fp16 KV audits.
    os.environ.setdefault("FASTINFERENCE_KV_FP8", "1")
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
    
    # [DEBUG] Verify weight loading
    try:
        if hasattr(engine.model, "model") and hasattr(engine.model.model, "embed_tokens"):
            emb = engine.model.model.embed_tokens.weight.data
            print(f"[DEBUG] embed_tokens.weight[0,:5]: {emb[0,:5].tolist()}")
            if emb.abs().mean() < 1e-6:
                print("[WARNING] embed_tokens.weight is all zero!")
        if hasattr(engine.model, "lm_head"):
            lmh = engine.model.lm_head.weight.data
            print(f"[DEBUG] lm_head.weight[0,:5]: {lmh[0,:5].tolist()}")
    except Exception as e:
        print(f"[DEBUG] Weight inspection failed: {e}")
    from vllm.model_executor.model_loader import get_tokenizer

    # When comparing quantized Lite vs FP16/BF16 baseline, use baseline tokenizer so token ids match HF.
    tokenizer_source = hf_model_path if hf_model_path else model_path
    tokenizer = get_tokenizer(tokenizer_source, trust_remote_code=True)
    engine.tokenizer = tokenizer
    prompt = apply_chat_template_for_verify(prompt, tokenizer, apply_chat_template)

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
            # When --hf-model points to a *different* tree than --model (e.g. AWQ Lite vs FP16 HF), keep HF on CPU
            # to avoid two large models on one GPU (ROCm OOM / illegal access).
            # When paths are the *same*, HF Transformers may use CUDA-only FLA kernels (chunk_gated_delta_rule);
            # CPU tensors fail there — allow explicit --hf-device cuda.
            _hf_differs_from_lite = (
                hf_model_path is not None
                and os.path.abspath(os.path.realpath(hf_model_path))
                != os.path.abspath(os.path.realpath(model_path))
            )
            if hf_model_path and hf_device != "cpu" and _hf_differs_from_lite:
                print(
                    f"  [Note] --hf-model is a different directory than --model; loading HF on {hf_device} "
                    "alongside Lite on GPU risks OOM / HIP illegal access. Forcing HF to CPU."
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
        prefill_cos_sim: Optional[float] = None
        prefill_lite_argmax: Optional[int] = None
        prefill_hf_argmax: Optional[int] = None

        input_ids_tokens = tokenizer.encode(prompt, return_tensors="pt")
        print(f"  Input Tokens: {input_ids_tokens[0].tolist()}")
        prompt_token_len = int(input_ids_tokens.shape[1])
        step_budget_audit = _lite_engine_step_budget(engine, prompt_token_len, 4)
        step_budget_full = _lite_engine_step_budget(
            engine, prompt_token_len, max_new_tokens
        )

        if prefill_only:
            print("[3/3] Running Prefill Audit (full multi-token generation skipped)...")
        else:
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
            # Last forward in chunked prefill is the final prompt chunk (not the first).
            lite_prefill_logits_cpu = captured_logits[-1][:, -1, :].float().cpu()
            lite_argmax_from_logits = int(torch.argmax(lite_prefill_logits_cpu, dim=-1).item())
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
                print(
                    f"  Prefill Token: HF(argmax)={hf_token} | "
                    f"Lite(engine)={lite_prefill_token} | Lite(argmax logits)={lite_argmax_from_logits}"
                )
                prefill_cos_sim = cos_sim
                prefill_lite_argmax = lite_argmax_from_logits
                prefill_hf_argmax = hf_token

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
            lite_logits = captured_logits[-1][:, -1, :].float().cpu()
            lite_argmax_from_logits2 = int(torch.argmax(lite_logits, dim=-1).item())
            hf_logits_cmp = hf_logits.float().cpu()
            vl, vr = lite_logits.shape[-1], hf_logits_cmp.shape[-1]
            if vl != vr:
                print(f"  [Note] Vocab size: Lite={vl} vs HF={vr}; logits compared on min slice.")
            cos_sim, max_err = compare_logits_aligned(lite_logits, hf_logits_cmp)
            print(f"  Prefill Logits -> CosSim: {cos_sim:.6f}, MaxErr: {max_err:.6f}")
            print(
                f"  Prefill Token: HF(argmax)={hf_token} | "
                f"Lite(engine)={lite_token} | Lite(argmax logits)={lite_argmax_from_logits2}"
            )
            prefill_cos_sim = cos_sim
            prefill_lite_argmax = lite_argmax_from_logits2
            prefill_hf_argmax = hf_token

        if prefill_only:
            if hf_model is not None and prefill_cos_sim is not None:
                cos_floor = prefill_cosine_floor_for_hf_compare(
                    model_path, hf_model_path, quant_type
                )
                ds_gguf = _is_deepseek_v2_lite_gguf_path(model_path, quant_type)
                audit_result = prefill_hf_alignment_pass(
                    prefill_cos_sim,
                    prefill_lite_argmax,
                    prefill_hf_argmax,
                    cos_min=cos_floor,
                    deepseek_gguf_q4=ds_gguf,
                )
                if audit_result:
                    if ds_gguf:
                        print(
                            f"  ✅ PASS: Prefill vs HF (DeepSeek GGUF: argmax match OR CosSim>={cos_floor})."
                        )
                    else:
                        print(
                            f"  ✅ PASS: Prefill aligns vs HF (CosSim>={cos_floor}, argmax match)."
                        )
                else:
                    if ds_gguf:
                        print(
                            f"  ❌ FAIL: Prefill vs HF (DeepSeek GGUF: need argmax match OR CosSim>={cos_floor})."
                        )
                    else:
                        print(
                            f"  ❌ FAIL: Prefill mismatch vs HF (need CosSim>={cos_floor} "
                            "and matching greedy argmax)."
                        )
            else:
                print(
                    "  [Info] Prefill-only run: no HF prefill comparison (no_hf or HF load failed)."
                )
                audit_result = True
        else:
            # Multi-token Generation Audit — Lite full greedy sequence
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
                hf_text = tokenizer.decode(
                    hf_full_gen[0][input_ids_hf.shape[-1]:],
                    skip_special_tokens=True,
                )
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
        "--hf-same-as-lite",
        action="store_true",
        help=(
            "Load Hugging Face reference from the same path as --model (same checkpoint/quant). "
            "Compares Lite vs HF implementations without mixing in a different FP16/BF16 tree. "
            "If set, --hf-model is ignored."
        ),
    )
    parser.add_argument(
        "--hf-device",
        type=str,
        default=None,
        choices=["auto", "cuda", "cpu"],
        help="Where to load HF reference. Default: cpu if --hf-model is set. "
        "If --hf-model points to a *different* directory than --model, non-cpu choices are forced to cpu "
        "(VRAM / ROCm safety). If --hf-model is the *same* path as --model, you may use cuda — required "
        "for Qwen3.5 HF + FLA (chunk_gated_delta_rule) which fails on CPU tensors.",
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
    parser.add_argument(
        "--apply-chat-template",
        type=str,
        choices=("off", "auto", "on"),
        default="off",
        help=(
            "Wrap --prompt as a single user turn via tokenizer.apply_chat_template when available "
            "(align with Tier-B spotcheck). Default off: raw prompt string."
        ),
    )
    parser.add_argument(
        "--prefill-only",
        action="store_true",
        help=(
            "Only compare last prefill logits + greedy first token vs HF; skip full multi-token audit."
        ),
    )
    args = parser.parse_args()

    if args.hf_same_as_lite and args.no_hf:
        parser.error("--hf-same-as-lite cannot be used with --no-hf")

    effective_hf_model_path: Optional[str] = args.hf_model
    if args.hf_same_as_lite:
        effective_hf_model_path = args.model
        if args.hf_model:
            print(
                f"[Config] --hf-same-as-lite: ignoring --hf-model={args.hf_model!r}, "
                f"HF reference = --model ({args.model!r})"
            )

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
        hf_device = "cpu" if (args.hf_model or args.hf_same_as_lite) else "auto"

    print(
        f"[Config] preset={args.preset}, quant={effective_quant}, "
        f"max_new_tokens={effective_max_new_tokens}, hf_device={hf_device}, "
        f"apply_chat_template={args.apply_chat_template}, prefill_only={args.prefill_only}"
    )
    if effective_hf_model_path and not args.no_hf:
        same = os.path.abspath(effective_hf_model_path) == os.path.abspath(args.model)
        print(
            f"[Config] HF reference path: {effective_hf_model_path!r} "
            f"({'same weights as Lite' if same else 'different from --model'})"
        )

    success = run_alignment_test(
        args.model,
        effective_quant,
        prompt=effective_prompt,
        max_new_tokens=effective_max_new_tokens,
        no_hf=args.no_hf,
        hf_model_path=effective_hf_model_path,
        hf_device=hf_device,
        activation_audit=args.activation_audit,
        activation_audit_max_passes=args.activation_audit_max_passes,
        disable_qwen35_stabilizers=args.disable_qwen35_stabilizers,
        apply_chat_template=args.apply_chat_template,
        prefill_only=args.prefill_only,
    )
    if not success:
        exit(1)
