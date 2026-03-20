# SPDX-License-Identifier: Apache-2.0
"""
Submodule-level alignment audit for layer 0 (HF vs Lite).

This narrows down where divergence begins inside layer 0 by comparing
intermediate outputs from matched submodules.

Fair comparison:
- AWQ Lite vs FP16 HF compares *different numerics* (quantized matmul vs dense);
  low CosSim on Linear/MLP is often expected.
- For *architecture / weight parity*, use ``--quant none`` and point ``--model``
  to the same (or equivalent) FP16/BF16 safetensors tree as ``--hf-model``.

Prefill vs decode: this script triggers one engine prefill; it does not validate
step-by-step decode cache paths (e.g. causal conv state).
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from vllm.config import CacheConfig, LoadConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.engine.lite_engine import LiteEngine
from vllm.sampling_params import SamplingParams


def _read_awq_group_size_and_bits(model_path: str) -> Tuple[int, int]:
    cfg_path = os.path.join(model_path, "config.json")
    group_size, bits = 128, 4
    if not os.path.isfile(cfg_path):
        return group_size, bits
    try:
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        qc = cfg.get("quantization_config") or {}
        groups = qc.get("config_groups")
        if isinstance(groups, dict):
            for g in groups.values():
                w = g.get("weights") if isinstance(g, dict) else None
                if isinstance(w, dict):
                    if w.get("group_size") is not None:
                        group_size = int(w["group_size"])
                    if w.get("num_bits") is not None:
                        bits = int(w["num_bits"])
                    break
    except Exception as e:
        print(f"[Warning] parse AWQ config failed: {e}")
    return group_size, bits


def _resolve_hf_dtype(hf_model_path: str) -> torch.dtype:
    try:
        p = os.path.join(hf_model_path, "config.json")
        with open(p, "r") as f:
            raw = json.load(f)
        tc = raw.get("text_config") or {}
        dtype_s = tc.get("dtype") if isinstance(tc, dict) else None
        if dtype_s is None and isinstance(tc, dict):
            dtype_s = tc.get("torch_dtype")
        if dtype_s is None:
            dtype_s = raw.get("torch_dtype")
        if isinstance(dtype_s, str) and "bfloat16" in dtype_s.lower():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


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
    return {
        "cos_sim": cos_sim,
        "mae": diff.mean().item(),
        "max_err": diff.max().item(),
    }


def _discover_layer0_base(root: nn.Module) -> str:
    """Return module path prefix for layer 0 (e.g. model.layers.0) from input_layernorm."""
    suffix = "layers.0.input_layernorm"
    for name, _mod in root.named_modules():
        if name == suffix or name.endswith("." + suffix):
            return name[: -len(".input_layernorm")]
    raise RuntimeError(
        "Cannot find layers.0.input_layernorm on model; unsupported backbone layout."
    )


def _discover_embed_tokens_path(root: nn.Module) -> Optional[str]:
    for name, mod in root.named_modules():
        if isinstance(mod, nn.Embedding) and name.endswith("embed_tokens"):
            return name
    return None


def _layer0_submodule_names(layer0_base: str) -> List[str]:
    return [
        f"{layer0_base}.input_layernorm",
        f"{layer0_base}.linear_attn.in_proj_qkv",
        f"{layer0_base}.linear_attn.conv1d",
        f"{layer0_base}.linear_attn.in_proj_z",
        f"{layer0_base}.linear_attn.out_proj",
        f"{layer0_base}.post_attention_layernorm",
        f"{layer0_base}.mlp.gate_proj",
        f"{layer0_base}.mlp.up_proj",
        f"{layer0_base}.mlp.down_proj",
        layer0_base,
    ]


def _register_named_hooks(root: torch.nn.Module, names: List[str], store: Dict[str, torch.Tensor]):
    modules = dict(root.named_modules())
    handles = []
    for name in names:
        m = modules.get(name)
        if m is None:
            continue

        def _make_hook(key):
            def _hook(_mod, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                if torch.is_tensor(t) and key not in store:
                    store[key] = t.detach().cpu()

            return _hook

        handles.append(m.register_forward_hook(_make_hook(name)))
    return handles


def _register_forward_pre_hooks_capture_input(
    root: nn.Module, names: List[str], store: Dict[str, torch.Tensor]
):
    """Capture first forward arg (e.g. hidden states before RMSNorm)."""
    modules = dict(root.named_modules())
    handles = []
    for name in names:
        m = modules.get(name)
        if m is None:
            continue

        def _make_pre(key):
            def _pre(_mod, inputs):
                if not inputs:
                    return
                t = inputs[0]
                if torch.is_tensor(t) and key not in store:
                    store[key] = t.detach().cpu()

            return _pre

        handles.append(m.register_forward_pre_hook(_make_pre(name)))
    return handles


_LAYER0_WEIGHT_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.out_proj.weight",
    "linear_attn.conv1d.weight",
    "linear_attn.in_proj_a.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.norm.weight",
    "linear_attn.A_log",
    "linear_attn.dt_bias",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def _report_layer0_weight_parity(
    hf_model: nn.Module,
    lite_model: nn.Module,
    hf_layer0_base: str,
    lite_layer0_base: str,
) -> None:
    hf_sd = hf_model.state_dict()
    lite_sd = lite_model.state_dict()
    print("[Weights] Layer0 parameter parity (HF vs Lite, float32):")
    any_missing = False
    for suf in _LAYER0_WEIGHT_SUFFIXES:
        kh = f"{hf_layer0_base}.{suf}"
        kl = f"{lite_layer0_base}.{suf}"
        if kh not in hf_sd or kl not in lite_sd:
            any_missing = True
            print(f"  {suf}: missing (hf={kh in hf_sd}, lite={kl in lite_sd})")
            continue
        a = hf_sd[kh].detach().float().cpu().reshape(-1)
        b = lite_sd[kl].detach().float().cpu().reshape(-1)
        if a.shape != b.shape:
            print(f"  {suf}: shape mismatch hf={tuple(hf_sd[kh].shape)} lite={tuple(lite_sd[kl].shape)}")
            continue
        diff = (a - b).abs()
        cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).item()
        print(
            f"  {suf}: CosSim={cos:.8f} max_abs={diff.max().item():.6e} mean_abs={diff.mean().item():.6e}"
        )
    if any_missing:
        print("[Weights] Some keys missing — architecture/key naming may differ from Qwen3.5 linear layer0.")


def _print_result_interpretation() -> None:
    print(
        "\n[Interpret] How to read the activation table:\n"
        "  • embed CosSim≈1: tokenizer + embedding weights match (good sanity check).\n"
        "  • input_layernorm <1: often CPU(FP32 RMS) HF vs CUDA(FP16) Lite rounding; try --hf-device cuda.\n"
        "  • linear_attn.*: pure attention path; gap vs HF = GatedDeltaNet/Conv1d/chunk-kernel numeric path.\n"
        "  • post_attention_layernorm & mlp.* OUTPUT hooks sit AFTER residual (x+attn). If attn differs,\n"
        "    inputs to these modules differ → CosSim can look catastrophic even when MLP *weights* match.\n"
        "  • Use --check-layer0-weights to verify tensors; use --compare-post-attn-residual to see x+attn\n"
        "    before post-norm (isolates residual-stream drift).\n"
        "  • Whole layer0 output CosSim (~0.9+) can still be acceptable while intermediate hooks look mixed.\n"
    )


def _build_lite_quant_config(quant: str, model_path: str):
    if quant == "none":
        return None
    if quant == "awq":
        from vllm.model_executor.layers.quantization.awq import AWQConfig

        gs, wb = _read_awq_group_size_and_bits(model_path)
        return AWQConfig(weight_bits=wb, group_size=gs)
    if quant == "gguf":
        from vllm.model_executor.layers.quantization.gguf import GGUFConfig

        return GGUFConfig()
    raise ValueError(f"Unknown quant mode: {quant}")


def _print_fairness_banner(quant: str, model_path: str, hf_model_path: str) -> None:
    mp_l = model_path.lower()
    hf_l = hf_model_path.lower()
    looks_awq = "awq" in mp_l or os.path.isfile(os.path.join(model_path, "quantize_config.json"))
    looks_fp16_hf = "fp16" in hf_l or "bf16" in hf_l or "awq" not in hf_l
    print(
        f"[Setup] Lite quant={quant} | model={model_path!r} | hf-model={hf_model_path!r}"
    )
    if quant == "awq" and looks_fp16_hf and looks_awq:
        print(
            "[Fairness] AWQ Lite vs FP16 HF: expect CosSim drops on quantized Linear/MLP; "
            "RMSNorm/embed should still be close if weights match. "
            "Use --quant none with the same dense checkpoint for both sides to test graph parity."
        )


def run_submodule_audit(
    model_path: str,
    hf_model_path: str,
    prompt: str,
    quant: str = "awq",
    include_embed: bool = False,
    hf_device: str = "cpu",
    check_layer0_weights: bool = False,
    compare_post_attn_residual: bool = False,
):
    os.environ["FASTINFERENCE_KV_FP8"] = "0"

    _print_fairness_banner(quant, model_path, hf_model_path)

    m_cfg = ModelConfig(model=model_path, tokenizer=model_path)
    c_cfg = CacheConfig(block_size=16, gpu_memory_utilization=0.9, swap_space=4)
    s_cfg = SchedulerConfig(max_num_batched_tokens=2048, max_num_seqs=32, max_model_len=2048)
    l_cfg = LoadConfig()
    q_cfg = _build_lite_quant_config(quant, model_path)
    v_cfg = VllmConfig(m_cfg, c_cfg, s_cfg, l_cfg, quant_config=q_cfg)
    lite_engine = LiteEngine(v_cfg)
    from vllm.model_executor.model_loader import get_tokenizer

    tokenizer = get_tokenizer(hf_model_path, trust_remote_code=True)
    lite_engine.tokenizer = tokenizer
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print(f"[Input] ids={input_ids[0].tolist()}")

    hf_dtype = _resolve_hf_dtype(hf_model_path)
    hf_cfg = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_path,
        config=hf_cfg,
        trust_remote_code=True,
        dtype=hf_dtype,
        low_cpu_mem_usage=True,
    ).eval()

    lite_root = lite_engine.model
    hf_layer0_base = _discover_layer0_base(hf_model)
    lite_layer0_base = _discover_layer0_base(lite_root)
    print(f"[Discover] HF layer0_base={hf_layer0_base!r} | Lite layer0_base={lite_layer0_base!r}")

    if check_layer0_weights:
        _report_layer0_weight_parity(hf_model, lite_root, hf_layer0_base, lite_layer0_base)

    hf_dev = torch.device("cpu")
    if hf_device == "cuda":
        if not torch.cuda.is_available():
            print("[Warning] --hf-device cuda requested but CUDA unavailable; using CPU for HF.")
        else:
            hf_dev = torch.device("cuda:0")
            print(
                f"[Setup] HF reference forward on {hf_dev} (reduces CPU-vs-GPU kernel drift vs Lite on CUDA)."
            )

    # Build hook name lists while HF is still on CPU (module graph unchanged).
    names = list(_layer0_submodule_names(hf_layer0_base))
    lite_names = list(_layer0_submodule_names(lite_layer0_base))
    if hf_layer0_base != lite_layer0_base:
        print(
            "[Warning] HF and Lite layer0 prefixes differ; comparing by parallel index "
            "(ensure both are layer 0 linear blocks)."
        )

    embed_hf = _discover_embed_tokens_path(hf_model)
    embed_lite = _discover_embed_tokens_path(lite_root)
    if include_embed:
        if embed_hf and embed_lite:
            names = [embed_hf] + names
            lite_names = [embed_lite] + lite_names
            print(f"[Discover] embed_tokens HF={embed_hf!r} Lite={embed_lite!r}")
        else:
            print(f"[Warning] --include-embed set but embed not found (hf={embed_hf}, lite={embed_lite})")

    # HF on GPU needs ~1x extra model VRAM; Lite is already resident — offload Lite weights to CPU
    # for the HF forward only, then move Lite back (KV cache stays on GPU).
    lite_on_gpu_for_hf_cuda = False
    lite_restore_dtype = torch.float16
    if hf_dev.type == "cuda":
        try:
            _p0 = next(lite_root.parameters())
            lite_on_gpu_for_hf_cuda = _p0.is_cuda
            lite_restore_dtype = _p0.dtype
        except StopIteration:
            lite_on_gpu_for_hf_cuda = False
        if lite_on_gpu_for_hf_cuda:
            print(
                "[Memory] Temporarily moving Lite model weights to CPU so HF can use GPU "
                "(avoids 2x model VRAM OOM; LiteEngine KV tensors stay on GPU)."
            )
            lite_root.cpu()
            torch.cuda.empty_cache()

    hf_model = hf_model.to(hf_dev)

    post_norm_name_hf = f"{hf_layer0_base}.post_attention_layernorm"
    post_norm_name_lite = f"{lite_layer0_base}.post_attention_layernorm"
    hf_pre: Dict[str, torch.Tensor] = {}
    lite_pre: Dict[str, torch.Tensor] = {}
    hf_pre_handles = []
    lite_pre_handles = []
    if compare_post_attn_residual:
        hf_pre_handles = _register_forward_pre_hooks_capture_input(
            hf_model, [post_norm_name_hf], hf_pre
        )
        lite_pre_handles = _register_forward_pre_hooks_capture_input(
            lite_root, [post_norm_name_lite], lite_pre
        )
        print(
            f"[Hooks] Forward-pre capturing input to post_attention_layernorm "
            f"(residual stream x+attn): HF={post_norm_name_hf!r} Lite={post_norm_name_lite!r}"
        )

    hf_store: Dict[str, torch.Tensor] = {}
    hf_handles = _register_named_hooks(hf_model, names, hf_store)
    attn_mask = torch.ones_like(input_ids, dtype=torch.long, device=hf_dev)
    with torch.inference_mode():
        _ = hf_model(input_ids.to(hf_dev), attention_mask=attn_mask)
    for h in hf_handles:
        h.remove()
    for h in hf_pre_handles:
        h.remove()

    hf_model = hf_model.cpu()
    torch.cuda.empty_cache()
    if lite_on_gpu_for_hf_cuda:
        _lite_dev = getattr(lite_engine, "device", torch.device("cuda:0"))
        print(f"[Memory] Moving Lite model weights back to {_lite_dev} for engine forward.")
        lite_root.to(device=_lite_dev, dtype=lite_restore_dtype)

    lite_store: Dict[str, torch.Tensor] = {}
    lite_handles = _register_named_hooks(lite_root, lite_names, lite_store)
    lite_engine.add_request("submod_audit", prompt, SamplingParams(max_tokens=1, temperature=0.0))
    while True:
        outs = lite_engine.step()
        if outs:
            break
    for h in lite_handles:
        h.remove()
    for h in lite_pre_handles:
        h.remove()

    if compare_post_attn_residual:
        h_pre = hf_pre.get(post_norm_name_hf)
        l_pre = lite_pre.get(post_norm_name_lite)
        if h_pre is not None and l_pre is not None:
            if h_pre.dim() >= 3 and l_pre.dim() >= 3:
                h_cmp, l_cmp = h_pre[:, -1], l_pre[:, -1]
            else:
                h_cmp, l_cmp = h_pre, l_pre
            pm = _pair_metrics(l_cmp, h_cmp)
            print(
                f"[Pre-norm residual] input to post_attention_layernorm "
                f"(x+attn): CosSim={pm['cos_sim']:.6f} MAE={pm['mae']:.6f} MaxErr={pm['max_err']:.6f}"
            )
        else:
            print(
                f"[Pre-norm residual] missing tensors "
                f"(hf={h_pre is not None}, lite={l_pre is not None})"
            )

    print(f"[Collected] HF={len(hf_store)} Lite={len(lite_store)}")
    print(
        "[Metrics] CosSim=cosine similarity vs HF (1.0=aligned), "
        "MAE=mean|Lite-HF|, MaxErr=max|Lite-HF|; compare last token when shape is [B,L,D]."
    )

    n_pairs = min(len(names), len(lite_names))
    for i in range(n_pairs):
        hf_name = names[i]
        lite_name = lite_names[i]
        h = hf_store.get(hf_name)
        l = lite_store.get(lite_name)
        label = f"{lite_name}  (vs HF {hf_name})"
        if h is None or l is None:
            print(f"{label}: missing (hf_tensor={h is not None}, lite_tensor={l is not None})")
            continue
        if h.dim() >= 3 and l.dim() >= 3:
            h_cmp = h[:, -1]
            l_cmp = l[:, -1]
        else:
            h_cmp = h
            l_cmp = l
        m = _pair_metrics(l_cmp, h_cmp)
        print(
            f"{label}: CosSim={m['cos_sim']:.6f} MAE={m['mae']:.6f} MaxErr={m['max_err']:.6f} "
            f"shape_hf={tuple(h.shape)} shape_lite={tuple(l.shape)}"
        )

    _print_result_interpretation()


if __name__ == "__main__":
    _EPILOG = r"""
Examples (paths are placeholders — use your local checkpoint dirs):

  # Strict structure/weight parity: same FP16 tree on both sides, Lite without AWQ.
  uv run python tests/verify_layer0_submodule_alignment.py \
    --model models/Qwen3.5-9B-FP16 --hf-model models/Qwen3.5-9B-FP16 \
    --quant none --include-embed

  # AWQ Lite vs FP16 HF reference (expect lower CosSim on quantized Linears).
  uv run python tests/verify_layer0_submodule_alignment.py \
    --model models/Qwen3.5-9B-AWQ --hf-model models/Qwen3.5-9B-FP16 \
    --quant awq
"""
    parser = argparse.ArgumentParser(
        description="HF vs Lite layer-0 submodule alignment (prefill). "
        "Use --quant none + same dense weights for strict parity checks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    parser.add_argument("--model", type=str, required=True, help="Lite checkpoint directory")
    parser.add_argument("--hf-model", type=str, required=True, help="HF reference directory")
    parser.add_argument(
        "--quant",
        type=str,
        default="awq",
        choices=["none", "awq", "gguf"],
        help="Lite quantization: 'none' loads dense fp weights only (fair vs FP16 HF).",
    )
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument(
        "--include-embed",
        action="store_true",
        help="Also hook token embeddings to see drift before layer0.",
    )
    parser.add_argument(
        "--hf-device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Run HF forward on this device. 'cuda' reduces mismatch vs Lite on GPU (needs extra VRAM).",
    )
    parser.add_argument(
        "--check-layer0-weights",
        action="store_true",
        help="Print CosSim/max_abs on layer0 parameters (confirms loader vs HF state_dict).",
    )
    parser.add_argument(
        "--compare-post-attn-residual",
        action="store_true",
        help="Compare tensor fed into post_attention_layernorm (x+attn) to see cascade from attention.",
    )
    args = parser.parse_args()
    run_submodule_audit(
        args.model,
        args.hf_model,
        args.prompt,
        quant=args.quant,
        include_embed=args.include_embed,
        hf_device=args.hf_device,
        check_layer0_weights=args.check_layer0_weights,
        compare_post_attn_residual=args.compare_post_attn_residual,
    )
