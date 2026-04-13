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
import re
import sys
import gc
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

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
    # Gemma4 31B Q4 (text-only bring-up; typically compressed-tensors/AWQ-style int4)
    "gemma4_31b_q4": {
        "prompt": "The capital of France is",
        "max_new_tokens": 10,
        "quant": "awq",
    },
    "gemma4_26b_a4b": {
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


def _read_model_config_json(model_path: str) -> dict[str, Any]:
    cfg_path = Path(model_path) / "config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _looks_like_gemma4_model_path(model_path: str) -> bool:
    try:
        raw = _read_model_config_json(model_path)
    except Exception:
        return False
    model_type = str(raw.get("model_type", "") or "").lower()
    text_model_type = str(((raw.get("text_config") or {}).get("model_type", "")) or "").lower()
    return model_type == "gemma4" or text_model_type.startswith("gemma4")


def _gemma4_map_ref_key(hf_key: str) -> Optional[str]:
    if hf_key.startswith("model.embed_vision.") or hf_key.startswith("model.audio_"):
        return None
    if hf_key.startswith("model.language_model."):
        name = "model." + hf_key[len("model.language_model.") :]
    else:
        name = hf_key
    # Gemma4 MoE checkpoints keep router/experts at layer root, while Lite model nests
    # them under decoder_layer.mlp.{router,experts}.
    m_exp = re.match(r"^model\.layers\.(\d+)\.experts\.(.+)$", name)
    if m_exp is not None:
        name = f"model.layers.{m_exp.group(1)}.mlp.experts.{m_exp.group(2)}"
    m_router = re.match(r"^model\.layers\.(\d+)\.router\.(.+)$", name)
    if m_router is not None:
        name = f"model.layers.{m_router.group(1)}.mlp.router.{m_router.group(2)}"
    if name.endswith(".weight_packed"):
        return name[: -len(".weight_packed")] + ".qweight"
    if name.endswith(".weight_scale"):
        return name[: -len(".weight_scale")] + ".scales"
    if name.endswith(".weight_shape"):
        return name
    if name.endswith(".router.per_expert_scale"):
        return name
    if name.endswith("_packed"):
        return name[: -len("_packed")] + ".qweight"
    if name.endswith("_scale"):
        return name[: -len("_scale")] + ".scales"
    if name.endswith("_zero"):
        return name[: -len("_zero")] + ".qzeros"
    if name.endswith("_shape"):
        return name[: -len("_shape")] + ".weight_shape"
    return name


def _set_module_attr_by_name(root: torch.nn.Module, full_name: str, value: Any) -> None:
    parts = full_name.split(".")
    obj: Any = root
    for part in parts[:-1]:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    leaf = parts[-1]
    if leaf.isdigit():
        obj[int(leaf)] = value
    else:
        setattr(obj, leaf, value)


def _resolve_gemma4_target_with_shared_mlp_fallback(
    model: torch.nn.Module, target: str
) -> Optional[str]:
    candidates = [target]
    m = re.match(
        r"^(model\.layers\.\d+\.mlp)\.(gate_proj|up_proj|down_proj)(\..+)$",
        target,
    )
    if m is not None:
        candidates.append(
            f"{m.group(1)}.shared_mlp.{m.group(2)}{m.group(3)}"
        )
    m_shared = re.match(
        r"^(model\.layers\.\d+\.mlp)\.shared_mlp\.(gate_proj|up_proj|down_proj)(\..+)$",
        target,
    )
    if m_shared is not None:
        candidates.append(f"{m_shared.group(1)}.{m_shared.group(2)}{m_shared.group(3)}")

    for cand in candidates:
        try:
            model.get_parameter(cand)
            return cand
        except Exception:
            continue
    return None


def _assign_gemma4_reference_weights(
    model: torch.nn.Module,
    hf_path: str,
    target_device: torch.device,
    *,
    safe_open_fn: Any,
) -> int:
    assigned = 0
    shards = sorted(str(p) for p in Path(hf_path).glob("*.safetensors"))
    for shard_path in shards:
        with safe_open_fn(shard_path, framework="pt", device="cpu") as f:
            for hf_key in f.keys():
                target = _gemma4_map_ref_key(hf_key)
                if target is None:
                    continue
                tensor = f.get_tensor(hf_key)
                if target.endswith(".weight_shape"):
                    module_name = target[: -len(".weight_shape")]
                    resolved_module_name = module_name
                    try:
                        model.get_submodule(module_name)
                    except Exception:
                        resolved_module_name = module_name
                    m2 = re.match(
                        r"^(model\.layers\.\d+\.mlp)\.(gate_proj|up_proj|down_proj)$",
                        module_name,
                    )
                    if m2 is not None:
                        alt = f"{m2.group(1)}.shared_mlp.{m2.group(2)}"
                        try:
                            model.get_submodule(alt)
                            resolved_module_name = alt
                        except Exception:
                            resolved_module_name = module_name
                    try:
                        module = model.get_submodule(resolved_module_name)
                    except Exception:
                        continue
                    module.weight_shape = tuple(int(x) for x in tensor.view(-1).tolist())
                    assigned += 1
                    continue
                resolved_target = _resolve_gemma4_target_with_shared_mlp_fallback(
                    model, target
                )
                if resolved_target is None:
                    continue
                current = model.get_parameter(resolved_target)
                if target.endswith((".qweight", ".scales", ".qzeros")):
                    tensor = tensor.to(device=target_device, dtype=tensor.dtype)
                else:
                    tensor = tensor.to(device=target_device, dtype=current.dtype)
                _set_module_attr_by_name(
                    model,
                    resolved_target,
                    torch.nn.Parameter(tensor.contiguous(), requires_grad=False),
                )
                assigned += 1
    if assigned == 0:
        raise RuntimeError(f"Gemma4 reference loader assigned no weights from {hf_path}")
    return assigned


class _Gemma4ReferenceWrapper(torch.nn.Module):
    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner.eval()

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        del attention_mask
        bsz, seqlen = input_ids.shape
        positions = (
            torch.arange(seqlen, device=input_ids.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(bsz, seqlen)
        )
        if input_ids.device.type == "cuda":
            block_size = 16
            max_model_len = max(256, ((seqlen + block_size - 1) // block_size) * block_size)
            num_blocks = max(1, max_model_len // block_size)
            num_layers = len(self.inner.model.layers)
            cfg = self.inner.model.config
            num_kv_heads = max(
                int(getattr(cfg, "num_key_value_heads", 1)),
                int(getattr(cfg, "num_global_key_value_heads", getattr(cfg, "num_key_value_heads", 1))),
            )
            kv_head_dim = max(
                int(getattr(cfg, "head_dim", 1)),
                int(getattr(cfg, "global_head_dim", getattr(cfg, "head_dim", 1))),
            )
            kv_caches = []
            kv_scale_caches = []
            for _ in range(num_layers):
                k = torch.zeros(
                    (num_blocks, block_size, num_kv_heads, kv_head_dim),
                    device=input_ids.device,
                    dtype=torch.uint8,
                )
                v = torch.zeros_like(k)
                kv_caches.append((k, v))
                ks = torch.zeros(
                    (num_blocks, block_size, num_kv_heads, 1),
                    device=input_ids.device,
                    dtype=torch.float32,
                )
                vs = torch.zeros_like(ks)
                kv_scale_caches.append((ks, vs))
            meta = {
                "slot_mapping": torch.arange(seqlen, device=input_ids.device, dtype=torch.long),
                "seq_lens": torch.tensor([seqlen], device=input_ids.device, dtype=torch.int32),
                "is_prefill": True,
                "kv_start_indices": torch.tensor([0], device=input_ids.device, dtype=torch.int32),
                "block_tables": torch.arange(num_blocks, device=input_ids.device, dtype=torch.int32).unsqueeze(0),
                "linear_attn_carry": [None] * num_layers,
                "linear_conv_carry": [None] * num_layers,
                "kv_scale_cache": kv_scale_caches,
                "kv_cache_dtype": "turbo_int4",
                "k_scale": 1.0,
                "v_scale": 1.0,
                "config": SimpleNamespace(kv_type="turbo_int4", k_scale=1.0, v_scale=1.0),
            }
        else:
            kv_caches = [None] * len(self.inner.model.layers)
            meta = {}
        logits = self.inner(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=meta,
        )
        return SimpleNamespace(logits=logits)


def _load_gemma4_reference_model(hf_path: str, dtype: torch.dtype, hf_device: str) -> torch.nn.Module:
    from safetensors import safe_open

    from vllm.config import CacheConfig, LoadConfig, ModelConfig, SchedulerConfig, VllmConfig
    from vllm.model_executor.layers.quantization.awq import AWQConfig
    from vllm.model_executor.models.gemma4 import Gemma4ForConditionalGeneration
    from vllm.transformers_utils.configs.gemma4 import build_fallback_hf_config

    raw = _read_model_config_json(hf_path)
    hf_config = build_fallback_hf_config(raw)
    quant_cfg = getattr(hf_config, "quantization_config", None)
    q_cfg = AWQConfig.from_config(quant_cfg) if isinstance(quant_cfg, dict) else None

    m_cfg = ModelConfig(model=hf_path, tokenizer=hf_path)
    m_cfg.hf_config = hf_config
    v_cfg = VllmConfig(
        m_cfg,
        CacheConfig(block_size=16, gpu_memory_utilization=0.1, swap_space=0),
        SchedulerConfig(max_num_batched_tokens=16, max_num_seqs=1, max_model_len=16),
        LoadConfig(),
        quant_config=q_cfg,
    )
    target_device = torch.device("cuda" if hf_device == "cuda" else "cpu")
    model = Gemma4ForConditionalGeneration(v_cfg).to(device=target_device, dtype=dtype).eval()

    _assign_gemma4_reference_weights(
        model,
        hf_path,
        target_device,
        safe_open_fn=safe_open,
    )
    return _Gemma4ReferenceWrapper(model)


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


def _extract_tensor_from_layer_output(output: Any) -> Optional[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    return None


def _get_gemma4_layers_for_hooks(model: torch.nn.Module) -> list[torch.nn.Module]:
    root: Any = model
    if hasattr(root, "inner"):
        root = root.inner
    if hasattr(root, "model") and hasattr(root.model, "layers"):
        return list(root.model.layers)
    if hasattr(root, "layers"):
        return list(root.layers)
    return []


def _register_last_token_layer_hooks(
    layers: list[torch.nn.Module],
    sink: Dict[int, torch.Tensor],
) -> list[Any]:
    handles: list[Any] = []
    for idx, layer in enumerate(layers):
        def _hook(_module: torch.nn.Module, _inputs: Any, output: Any, i: int = idx) -> None:
            t = _extract_tensor_from_layer_output(output)
            if t is None or t.ndim < 3:
                return
            sink[i] = t[:, -1, :].detach().float().cpu()

        handles.append(layer.register_forward_hook(_hook))
    return handles


def _print_first_drift_layer(
    lite_layer_last_token: Dict[int, torch.Tensor],
    hf_layer_last_token: Dict[int, torch.Tensor],
    cos_threshold: float,
) -> None:
    shared_layers = sorted(set(lite_layer_last_token.keys()) & set(hf_layer_last_token.keys()))
    if not shared_layers:
        print("  [LayerDrift] no shared layer captures; skipping.")
        return

    first_drift: Optional[int] = None
    first_drift_cos: Optional[float] = None
    for li in shared_layers:
        cos_sim, _ = compare_logits_aligned(
            lite_layer_last_token[li],
            hf_layer_last_token[li],
        )
        if not math.isfinite(cos_sim):
            continue
        if cos_sim < cos_threshold:
            first_drift = li
            first_drift_cos = cos_sim
            break

    if first_drift is None:
        print(
            f"  [LayerDrift] compared={len(shared_layers)} first_drift_layer=None "
            f"(all cos>={cos_threshold:.4f})"
        )
    else:
        print(
            f"  [LayerDrift] compared={len(shared_layers)} first_drift_layer={first_drift} "
            f"cos={first_drift_cos:.6f} threshold={cos_threshold:.4f}"
        )

def _load_hf_reference_model(hf_path: str, dtype: torch.dtype, hf_device: str):
    """
    Load HF reference; supports dense CausalLM and Qwen3.5 multimodal checkpoints.
    Use hf_device='cpu' when Lite already occupies GPU VRAM to avoid OOM / illegal memory access.
    """
    if _looks_like_gemma4_model_path(hf_path):
        return _load_gemma4_reference_model(hf_path, dtype, hf_device)

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
    max_model_len: int = 2048,
    max_num_batched_tokens: int = 2048,
    gpu_memory_utilization: float = 0.9,
    no_hf=False,
    hf_model_path=None,
    hf_device="auto",
    activation_audit: bool = False,
    activation_audit_max_passes: Optional[int] = None,
    disable_qwen35_stabilizers: bool = False,
    apply_chat_template: str = "off",
    prefill_only: bool = False,
    awq_force_fused: bool = False,
    awq_disable_fused: bool = False,
    report_first_drift_layer: bool = False,
    drift_cos_threshold: float = 0.995,
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

    # Default to TurboQuant INT4 KV unless caller explicitly requests otherwise.
    # Keep legacy FASTINFERENCE_KV_FP8 behavior only when KV_TYPE is explicitly "auto".
    if "FASTINFERENCE_KV_TYPE" not in os.environ and "FASTINFERENCE_KV_FP8" not in os.environ:
        os.environ["FASTINFERENCE_KV_TYPE"] = "turbo_int4"
    if quant_type == "awq":
        if awq_disable_fused:
            os.environ["FASTINFERENCE_AWQ_FUSED_GEMM"] = "0"
            os.environ["FASTINFERENCE_AWQ_FUSED_GEMM_FORCE"] = "0"
            print("  [AWQ] fused GEMM disabled for this audit.")
        elif awq_force_fused:
            os.environ["FASTINFERENCE_AWQ_FUSED_GEMM"] = "1"
            os.environ["FASTINFERENCE_AWQ_FUSED_GEMM_FORCE"] = "1"
            print("  [AWQ] fused GEMM force-enabled for this audit.")
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
    c_cfg = CacheConfig(block_size=16, gpu_memory_utilization=gpu_memory_utilization, swap_space=4)
    s_cfg = SchedulerConfig(
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=32,
        max_model_len=max_model_len,
    )
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
        lite_layer_last_token: Dict[int, torch.Tensor] = {}
        hf_layer_last_token: Dict[int, torch.Tensor] = {}

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
            lite_handles: list[Any] = []
            if report_first_drift_layer and _looks_like_gemma4_model_path(model_path):
                lite_layers = _get_gemma4_layers_for_hooks(engine.model)
                lite_handles = _register_last_token_layer_hooks(lite_layers, lite_layer_last_token)

            def audit_forward(*args, **kwargs):
                if lite_handles:
                    lite_layer_last_token.clear()
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
            for h in lite_handles:
                h.remove()
            # Last forward in chunked prefill is the final prompt chunk (not the first).
            lite_prefill_logits_cpu = captured_logits[-1][:, -1, :].float().cpu()
            lite_argmax_from_logits = int(torch.argmax(lite_prefill_logits_cpu, dim=-1).item())
            if not torch.isfinite(lite_prefill_logits_cpu).all():
                n_bad = int((~torch.isfinite(lite_prefill_logits_cpu)).sum().item())
                print(f"  [Warning] Lite prefill logits: {n_bad} non-finite values (check AWQ group_size / weights).")

            print(f"  [2b] Loading HF Reference (PyTorch {dtype}) from {hf_load_path}...")
            if (
                hf_device == "cuda"
                and prefill_only
                and _looks_like_gemma4_model_path(hf_load_path)
            ):
                print("  [Info] Releasing LiteEngine GPU state before Gemma4 CUDA reference load.")
                del engine
                gc.collect()
                torch.cuda.empty_cache()
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
                hf_handles: list[Any] = []
                if report_first_drift_layer and _looks_like_gemma4_model_path(hf_load_path):
                    hf_layers = _get_gemma4_layers_for_hooks(hf_model)
                    hf_handles = _register_last_token_layer_hooks(hf_layers, hf_layer_last_token)
                    hf_layer_last_token.clear()
                with torch.inference_mode():
                    hf_outputs = hf_model(input_ids_hf, attention_mask=attn_mask)
                    if hasattr(hf_outputs, "logits") and hf_outputs.logits is not None:
                        hf_logits = hf_outputs.logits[:, -1, :]
                    elif isinstance(hf_outputs, tuple):
                        hf_logits = hf_outputs[0][:, -1, :]
                    else:
                        raise RuntimeError("HF model output has no usable logits for audit.")
                    hf_token = torch.argmax(hf_logits, dim=-1).item()
                for h in hf_handles:
                    h.remove()
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
            hf_handles: list[Any] = []
            if report_first_drift_layer and _looks_like_gemma4_model_path(hf_load_path):
                hf_layers = _get_gemma4_layers_for_hooks(hf_model)
                hf_handles = _register_last_token_layer_hooks(hf_layers, hf_layer_last_token)
                hf_layer_last_token.clear()
            with torch.inference_mode():
                hf_outputs = hf_model(input_ids_hf, attention_mask=attn_mask)
                if hasattr(hf_outputs, "logits") and hf_outputs.logits is not None:
                    hf_logits = hf_outputs.logits[:, -1, :]
                elif isinstance(hf_outputs, tuple):
                    hf_logits = hf_outputs[0][:, -1, :]
                else:
                    raise RuntimeError("HF model output has no usable logits for audit.")
                hf_token = torch.argmax(hf_logits, dim=-1).item()
            for h in hf_handles:
                h.remove()

            captured_logits = []
            original_forward = engine.model.forward
            lite_handles: list[Any] = []
            if report_first_drift_layer and _looks_like_gemma4_model_path(model_path):
                lite_layers = _get_gemma4_layers_for_hooks(engine.model)
                lite_handles = _register_last_token_layer_hooks(lite_layers, lite_layer_last_token)

            def audit_forward(*args, **kwargs):
                if lite_handles:
                    lite_layer_last_token.clear()
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
            for h in lite_handles:
                h.remove()
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
            if (
                report_first_drift_layer
                and hf_model is not None
                and lite_layer_last_token
                and hf_layer_last_token
            ):
                _print_first_drift_layer(
                    lite_layer_last_token,
                    hf_layer_last_token,
                    cos_threshold=drift_cos_threshold,
                )
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
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.environ.get("FASTINFERENCE_VERIFY_MAX_MODEL_LEN", "2048")),
        help="Override Lite scheduler max_model_len for the audit.",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=int(os.environ.get("FASTINFERENCE_VERIFY_MAX_BATCHED_TOKENS", "2048")),
        help="Override Lite scheduler max_num_batched_tokens for the audit.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.environ.get("FASTINFERENCE_VERIFY_GPU_MEM_UTIL", "0.9")),
        help="Override Lite cache gpu_memory_utilization for the audit.",
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
    parser.add_argument(
        "--awq-force-fused",
        action="store_true",
        help="Force-enable AWQ fused GEMM path (sets FASTINFERENCE_AWQ_FUSED_GEMM=1 and FORCE=1).",
    )
    parser.add_argument(
        "--awq-disable-fused",
        action="store_true",
        help="Force-disable AWQ fused GEMM path for A/B regression checks.",
    )
    parser.add_argument(
        "--report-first-drift-layer",
        action="store_true",
        help=(
            "Prefill-only diagnostic: for Gemma4, compare Lite/HF last-token hidden states per layer "
            "and print the first layer whose cosine drops below --drift-cos-threshold."
        ),
    )
    parser.add_argument(
        "--drift-cos-threshold",
        type=float,
        default=0.995,
        help="Cosine threshold used by --report-first-drift-layer.",
    )
    args = parser.parse_args()

    if args.hf_same_as_lite and args.no_hf:
        parser.error("--hf-same-as-lite cannot be used with --no-hf")
    if args.awq_force_fused and args.awq_disable_fused:
        parser.error("--awq-force-fused and --awq-disable-fused are mutually exclusive")

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
        f"apply_chat_template={args.apply_chat_template}, prefill_only={args.prefill_only}, "
        f"report_first_drift_layer={args.report_first_drift_layer}, "
        f"max_model_len={args.max_model_len}, max_num_batched_tokens={args.max_num_batched_tokens}, "
        f"gpu_mem_util={args.gpu_memory_utilization}"
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
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        no_hf=args.no_hf,
        hf_model_path=effective_hf_model_path,
        hf_device=hf_device,
        activation_audit=args.activation_audit,
        activation_audit_max_passes=args.activation_audit_max_passes,
        disable_qwen35_stabilizers=args.disable_qwen35_stabilizers,
        apply_chat_template=args.apply_chat_template,
        prefill_only=args.prefill_only,
        awq_force_fused=args.awq_force_fused,
        awq_disable_fused=args.awq_disable_fused,
        report_first_drift_layer=args.report_first_drift_layer,
        drift_cos_threshold=args.drift_cos_threshold,
    )
    if not success:
        exit(1)
