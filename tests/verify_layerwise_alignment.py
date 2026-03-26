# SPDX-License-Identifier: Apache-2.0
"""
Layer-wise hidden-state alignment audit between LitevLLM and HF baseline.

Usage example:
uv run python tests/verify_layerwise_alignment.py \
  --model models/Qwen3.5-9B-AWQ \
  --hf-model models/Qwen3.5-9B-FP16 \
  --quant awq
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from vllm.config import CacheConfig, LoadConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.engine.lite_engine import LiteEngine
from vllm.sampling_params import SamplingParams


def _read_awq_group_size_and_bits(model_path: str) -> Tuple[int, int]:
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
        print(f"[Warning] Failed to parse AWQ config: {e}")
    return group_size, bits


def _resolve_hf_dtype(hf_model_path: str) -> torch.dtype:
    try:
        p = os.path.join(hf_model_path, "config.json")
        with open(p, "r") as f:
            raw = json.load(f)
        tc = raw.get("text_config") or {}
        dtype_s = None
        if isinstance(tc, dict):
            dtype_s = tc.get("dtype") or tc.get("torch_dtype")
        if dtype_s is None:
            dtype_s = raw.get("torch_dtype")
        if isinstance(dtype_s, str) and "bfloat16" in dtype_s.lower():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


def _load_hf_model(hf_model_path: str, dtype: torch.dtype):
    cfg = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
    common = dict(
        pretrained_model_name_or_path=hf_model_path,
        config=cfg,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    try:
        return AutoModelForCausalLM.from_pretrained(**common).eval().to("cpu")
    except Exception:
        return AutoModel.from_pretrained(**common).eval().to("cpu")


def _flatten_finite_pair(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x = a.float().reshape(-1)
    y = b.float().reshape(-1)
    n = min(x.numel(), y.numel())
    x = x[:n]
    y = y[:n]
    mask = torch.isfinite(x) & torch.isfinite(y)
    if not mask.any():
        return x[:0], y[:0]
    return x[mask], y[mask]


def _pair_metrics(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    x, y = _flatten_finite_pair(a, b)
    if x.numel() == 0:
        return {"cos_sim": float("nan"), "mae": float("nan"), "max_err": float("nan")}
    cos_sim = F.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0), dim=1).squeeze(0).item()
    diff = (x - y).abs()
    return {"cos_sim": cos_sim, "mae": diff.mean().item(), "max_err": diff.max().item()}


def _resolve_hf_text_backbone(hf_model):
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
        return hf_model.model
    for name in ("language_model", "text_model", "lm", "transformer"):
        if hasattr(hf_model, name):
            obj = getattr(hf_model, name)
            if hasattr(obj, "layers"):
                return obj
            if hasattr(obj, "model") and hasattr(obj.model, "layers"):
                return obj.model
    raise RuntimeError("Cannot locate HF text backbone with .layers")


def run_layerwise_audit(
    model_path: str,
    hf_model_path: str,
    quant: str,
    prompt: str,
):
    os.environ.setdefault("FASTINFERENCE_KV_FP8", "1")

    # 1) Lite engine
    m_cfg = ModelConfig(model=model_path, tokenizer=model_path)
    c_cfg = CacheConfig(block_size=16, gpu_memory_utilization=0.9, swap_space=4)
    s_cfg = SchedulerConfig(max_num_batched_tokens=2048, max_num_seqs=32, max_model_len=2048)
    l_cfg = LoadConfig()
    q_cfg = None
    if quant == "awq":
        from vllm.model_executor.layers.quantization.awq import AWQConfig

        gs, wb = _read_awq_group_size_and_bits(model_path)
        q_cfg = AWQConfig(weight_bits=wb, group_size=gs)
        print(f"[Lite] AWQ parsed: group_size={gs}, weight_bits={wb}")
    elif quant == "gguf":
        from vllm.model_executor.layers.quantization.gguf import GGUFConfig

        q_cfg = GGUFConfig()
    v_cfg = VllmConfig(m_cfg, c_cfg, s_cfg, l_cfg, quant_config=q_cfg)
    engine = LiteEngine(v_cfg)

    from vllm.model_executor.model_loader import get_tokenizer

    tokenizer = get_tokenizer(hf_model_path, trust_remote_code=True)
    engine.tokenizer = tokenizer
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print(f"[Input] ids={input_ids[0].tolist()}")

    # 2) HF forward hooks
    hf_dtype = _resolve_hf_dtype(hf_model_path)
    print(f"[HF] loading from {hf_model_path}, dtype={hf_dtype}, device=cpu")
    hf_model = _load_hf_model(hf_model_path, hf_dtype)
    hf_backbone = _resolve_hf_text_backbone(hf_model)

    hf_layer_outs: List[torch.Tensor] = []
    hf_handles = []
    for i, layer in enumerate(hf_backbone.layers):
        def _make_hook(idx):
            def _hook(_mod, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                hf_layer_outs.append(t.detach().cpu())
            return _hook

        hf_handles.append(layer.register_forward_hook(_make_hook(i)))

    input_ids_hf = input_ids.to("cpu")
    attn = torch.ones_like(input_ids_hf, dtype=torch.long)
    with torch.inference_mode():
        _ = hf_model(input_ids_hf, attention_mask=attn)
    for h in hf_handles:
        h.remove()

    # 3) Lite forward hooks (first prefill forward)
    lite_layer_outs: List[torch.Tensor] = []
    lite_handles = []
    lite_backbone = engine.model.model if hasattr(engine.model, "model") else engine.model
    for i, layer in enumerate(lite_backbone.layers):
        def _make_hook(idx):
            def _hook(_mod, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                lite_layer_outs.append(t.detach().cpu())
            return _hook

        lite_handles.append(layer.register_forward_hook(_make_hook(i)))

    engine.add_request("layer_audit", prompt, SamplingParams(max_tokens=1, temperature=0.0))
    while True:
        outs = engine.step()
        if outs:
            break
    for h in lite_handles:
        h.remove()

    n = min(len(hf_layer_outs), len(lite_layer_outs))
    print(f"[Audit] compared_layers={n} (HF={len(hf_layer_outs)}, Lite={len(lite_layer_outs)})")
    print("[Metrics] CosSim=cosine similarity vs HF (1.0=aligned); per layer uses last token hidden.")
    first_bad: Optional[int] = None
    for i in range(n):
        h_t = hf_layer_outs[i][:, -1, :]
        l_t = lite_layer_outs[i][:, -1, :]
        m = _pair_metrics(l_t, h_t)
        print(
            f"layer.{i:02d} CosSim={m['cos_sim']:.6f} MAE={m['mae']:.6f} MaxErr={m['max_err']:.6f}"
        )
        if first_bad is None and (
            not torch.isfinite(torch.tensor(m["cos_sim"])) or m["cos_sim"] < 0.95
        ):
            first_bad = i
    if first_bad is None:
        print("[Result] No obvious bad layer (CosSim < 0.95).")
    else:
        print(f"[Result] First divergent layer: {first_bad}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--hf-model", type=str, required=True)
    parser.add_argument("--quant", type=str, default="awq", choices=["none", "awq", "gguf"])
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    args = parser.parse_args()

    run_layerwise_audit(
        model_path=args.model,
        hf_model_path=args.hf_model,
        quant=args.quant,
        prompt=args.prompt,
    )
