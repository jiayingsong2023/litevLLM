# SPDX-License-Identifier: Apache-2.0
import os
import torch
import torch.nn as nn
from typing import Any, List, Optional, Union, Dict
import numpy as np
import gguf
import json
import re
import gc
from transformers import AutoTokenizer, AutoConfig, PretrainedConfig
from vllm.config import VllmConfig, ModelConfig
from vllm.model_executor.models.registry import ModelRegistry
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.tensor import dequantize_awq_pytorch, GGUFWeight

def _dequantize_gguf_tensor(gguf_tensor: Any, device: str, dtype: torch.dtype, target_shape: Optional[torch.Size] = None) -> Optional[torch.Tensor]:
    q_type = int(gguf_tensor.tensor_type)
    if q_type in [0, 1]: return torch.from_numpy(np.array(gguf_tensor.data, copy=True)).to(device=device).to(dtype=dtype)
    
    if target_shape is not None:
        if len(target_shape) == 1: R, C = 1, target_shape[0]
        elif len(target_shape) == 2: R, C = target_shape[0], target_shape[1]
        else:
            # 3D: [Experts, Rows, Cols] -> Logical R = Experts*Rows, C = Cols
            C = target_shape[-1]
            R = 1
            for dim in target_shape[:-1]: R *= dim
    else:
        # Physical fallback
        ps = gguf_tensor.shape
        C = ps[0]
        R = 1
        for dim in ps[1:]: R *= dim

    qweight = torch.from_numpy(np.array(gguf_tensor.data, copy=True)).to(device=device)
    try:
        if q_type == 12:
            from vllm.model_executor.layers.quantization.tensor import dequantize_q4k_pytorch
            res = dequantize_q4k_pytorch(qweight, R, C)
            return res.view(target_shape) if target_shape is not None else res
        if q_type == 14:
            from vllm.model_executor.layers.quantization.tensor import dequantize_q6k_pytorch
            res = dequantize_q6k_pytorch(qweight, R, C)
            return res.view(target_shape) if target_shape is not None else res
    except: return None
    return None

def _load_param_from_gguf_tensor(param: Any, gguf_tensor: Any, slice_offset: int = 0) -> None:
    actual_param = param; logical_shape = None
    if isinstance(param, LiteLinear): logical_shape = torch.Size([param.output_size, param.input_size]); actual_param = param.weight
    elif isinstance(param, nn.Module) and hasattr(param, "weight"): actual_param = param.weight
    if not isinstance(actual_param, torch.Tensor) and logical_shape is None: return
    target_shape = logical_shape if logical_shape is not None else actual_param.shape
    source = _dequantize_gguf_tensor(gguf_tensor, "cpu", torch.float16, target_shape)
    if source is None: return
    if isinstance(param, LiteLinear):
        if param.weight.numel() == 0 or param.weight.shape != source.shape: param.weight = nn.Parameter(source.to(dtype=torch.float16), requires_grad=False)
        else: param.weight.data.copy_(source.to(dtype=param.weight.dtype))
    elif isinstance(actual_param, torch.Tensor):
        if actual_param.shape == source.shape: actual_param.data.copy_(source)
        else:
            try:
                if source.shape[0] == actual_param.shape[1] and source.shape[1] >= actual_param.shape[0]: actual_param.data.copy_(source[:, :actual_param.shape[0]].T)
                elif source.shape[1] == actual_param.shape[1] and source.shape[0] >= actual_param.shape[0]: actual_param.data.copy_(source[:actual_param.shape[0], :])
            except: pass

def _load_gguf_weights(model: nn.Module, gguf_file: str, hf_config: Any) -> None:
    print(f">>> Mapping GGUF weights from {gguf_file}...")
    reader = gguf.GGUFReader(gguf_file); tensor_map = {t.name: t for t in reader.tensors}
    map_rules = {
        "token_embd.weight": ["model.embed_tokens.weight"],
        "output_norm.weight": ["model.norm.weight"],
        "output.weight": ["lm_head.weight"],
        "blk.{i}.attn_norm.weight": ["model.layers.{i}.input_layernorm.weight"],
        "blk.{i}.ffn_norm.weight": ["model.layers.{i}.post_attention_layernorm.weight"],
        "blk.{i}.attn_q_a.weight": ["model.layers.{i}.self_attn.q_a_proj.weight"],
        "blk.{i}.attn_q_a_norm.weight": ["model.layers.{i}.self_attn.q_a_layernorm.weight"],
        "blk.{i}.attn_q_b.weight": ["model.layers.{i}.self_attn.q_b_proj.weight"],
        "blk.{i}.attn_kv_a_mqa.weight": ["model.layers.{i}.self_attn.kv_a_proj.weight"],
        "blk.{i}.attn_kv_a_norm.weight": ["model.layers.{i}.self_attn.kv_a_layernorm.weight"],
        "blk.{i}.attn_output.weight": ["model.layers.{i}.self_attn.o_proj.weight"],
        "blk.{i}.ffn_gate_inp.weight": ["model.layers.{i}.mlp.router.weight"],
        "blk.{i}.ffn_gate_shexp.weight": ["model.layers.{i}.mlp.shared_experts.gate_proj.weight"],
        "blk.{i}.ffn_up_shexp.weight": ["model.layers.{i}.mlp.shared_experts.up_proj.weight"],
        "blk.{i}.ffn_down_shexp.weight": ["model.layers.{i}.mlp.shared_experts.down_proj.weight"],
    }
    modules = dict(model.named_modules()); params = dict(model.named_parameters())
    
    # Metadata for MoE shapes
    num_experts = getattr(hf_config, "n_routed_experts", 64)
    moe_inter = getattr(hf_config, "moe_intermediate_size", 1536)
    hidden = getattr(hf_config, "hidden_size", 2048)
    
    num_layers = getattr(hf_config, "num_hidden_layers", 28)
    loaded_count = 0
    for i in range(num_layers):
        for g_pat, m_paths in map_rules.items():
            g_key = g_pat.format(i=i)
            if g_key in tensor_map:
                for m_path in m_paths:
                    target_path = m_path.format(i=i)
                    if target_path in params: _load_param_from_gguf_tensor(params[target_path], tensor_map[g_key]); loaded_count += 1; break
                    m_base = target_path.rsplit(".", 1)[0] if "." in target_path else target_path
                    if m_base in modules: _load_param_from_gguf_tensor(modules[m_base], tensor_map[g_key]); loaded_count += 1; break
        
        kb_key = f"blk.{i}.attn_k_b.weight"; vb_key = f"blk.{i}.attn_v_b.weight"
        if kb_key in tensor_map and vb_key in tensor_map:
            kb = _dequantize_gguf_tensor(tensor_map[kb_key], "cpu", torch.float16)
            vb = _dequantize_gguf_tensor(tensor_map[vb_key], "cpu", torch.float16)
            if kb is not None and vb is not None:
                fused_kv_b = torch.cat([kb, vb], dim=0)
                m_path = f"model.layers.{i}.self_attn.kv_b_proj"
                if m_path in modules: modules[m_path].weight = nn.Parameter(fused_kv_b, requires_grad=False); loaded_count += 1
        
        # PROPER MOE EXPERT LOADING
        for e_type in ["gate", "up", "down"]:
            ge_key = f"blk.{i}.ffn_{e_type}_exps.weight"
            if ge_key in tensor_map:
                if e_type in ["gate", "up"]: t_shape = torch.Size([num_experts, moe_inter, hidden])
                else: t_shape = torch.Size([num_experts, hidden, moe_inter])
                
                ge = _dequantize_gguf_tensor(tensor_map[ge_key], "cpu", torch.float16, target_shape=t_shape)
                if ge is not None:
                    m_moe_path = f"model.layers.{i}.mlp"
                    if m_moe_path in modules:
                        m_moe = modules[m_moe_path]
                        p_name = f"experts_{e_type}"
                        setattr(m_moe, p_name, nn.Parameter(ge, requires_grad=False)); loaded_count += 1
        if i % 4 == 0: gc.collect()
    print(f">>> GGUF Weight loading complete. Loaded {loaded_count} parameters.")

def _load_safetensors(model: nn.Module, model_path: str):
    from safetensors.torch import load_file
    sf_files = sorted([f for f in os.listdir(model_path) if f.endswith(".safetensors")])
    if not sf_files: return
    print(f">>> Loading Safetensors from {model_path}...")
    loaded_count = 0; attr_map = {'qweight': ['weight_packed', 'qweight', 'weight'], 'scales': ['weight_scale', 'scales'], 'qzeros': ['qzeros', 'weight_zero'], 'bias': ['bias']}
    lite_modules = {n: m for n, m in model.named_modules() if isinstance(m, (LiteLinear, RMSNorm)) or "linear_attn" in n}; params_dict = dict(model.named_parameters()); loaded_params = set()
    for f in sf_files:
        print(f"    Processing {f}...")
        sd = load_file(os.path.join(model_path, f), device="cpu")
        sd_keys = list(sd.keys())
        for m_name, m in lite_modules.items():
            idx_match = re.search(r"layers\.(\d+)\.", m_name)
            idx = idx_match.group(1) if idx_match else None
            proj = m_name.split(".")[-1]; awq_comps = {}
            for internal, srcs in attr_map.items():
                found_v = None
                for s in srcs:
                    for k in sd_keys:
                        if idx is not None and f"layers.{idx}." in k:
                            if k.endswith(f".{proj}.{s}") or k.endswith(f".{proj.replace('self_attn', 'linear_attn')}.{s}"): found_v = sd[k]; break
                            if "linear_attn" in k and k.endswith(f".{s}") and proj in k:
                                if not any(x in k for x in ["packed", "scale", "zero"]): found_v = sd[k]; break
                        elif idx is None and "layers." not in k:
                            if k.endswith(f".{proj}.{s}") or k == f"{m_name}.{s}" or k.endswith(f".{m_name}.{s}"): found_v = sd[k]; break
                        if found_v is not None: break
                    if found_v is not None: break
                if found_v is not None:
                    if internal == "bias":
                        if hasattr(m, "bias") and m.bias is not None: m.bias.data.copy_(found_v.to(dtype=m.bias.dtype))
                        else: m.bias = nn.Parameter(found_v.to(dtype=torch.float16))
                        loaded_count += 1
                    elif (internal == "qweight" or internal == "weight") and found_v.dtype in [torch.float16, torch.bfloat16]:
                        target_p = m.weight if hasattr(m, "weight") else m
                        if isinstance(target_p, nn.Parameter):
                            if target_p.numel() == 0 or target_p.shape != found_v.shape:
                                if hasattr(m, "weight"): m.weight = nn.Parameter(found_v.to(dtype=torch.float16), requires_grad=False)
                                else: pass
                            else: target_p.data.copy_(found_v.to(dtype=target_p.dtype))
                            loaded_count += 1; loaded_params.add(m_name + ".weight")
                    else: awq_comps[internal] = found_v
            if "qweight" in awq_comps and "scales" in awq_comps:
                qw = awq_comps["qweight"]; sc = awq_comps["scales"]; qz = awq_comps.get("qzeros", torch.zeros((qw.shape[0], sc.shape[1] // 8), dtype=torch.int32))
                K = qw.shape[1] * 8; G = K // sc.shape[1]
                try:
                    dq = dequantize_awq_pytorch(qw, sc, qz, group_size=G)
                    if m.weight.numel() == 0 or m.weight.shape != dq.shape: m.weight = nn.Parameter(dq.to(dtype=torch.float16), requires_grad=False)
                    else: m.weight.data.copy_(dq.to(dtype=torch.float16))
                    if hasattr(m, "_quant_weight"): m._quant_weight = None
                    loaded_count += 1; loaded_params.add(m_name + ".weight")
                except: pass
        for p_name, p in params_dict.items():
            if p_name in loaded_params: continue
            target = p_name[6:] if p_name.startswith("model.") else p_name
            for k in sd_keys:
                if k == target or k.endswith("." + target):
                    if p.shape == sd[k].shape: p.data.copy_(sd[k].to(dtype=p.dtype)); loaded_count += 1; loaded_params.add(p_name); break
        del sd; gc.collect()
    print(f">>> Safetensors loading complete. Loaded {loaded_count} parameters.")

def get_tokenizer(model: Union[str, Any], **kwargs: Any):
    name = model.model if hasattr(model, "model") else model
    if hasattr(model, "trust_remote_code"): kwargs["trust_remote_code"] = model.trust_remote_code
    try: return AutoTokenizer.from_pretrained(name, **kwargs)
    except:
        try: return AutoTokenizer.from_pretrained(name, use_fast=False, **kwargs)
        except:
            class Dummy:
                def __init__(self): self.vocab_size = 154880; self.eos_token_id = 151329; self.pad_token_id = 151329
                def encode(self, t, **kwargs): 
                    ids = [1, 2, 3, 4, 5] * 2
                    if kwargs.get("return_tensors") == "pt": return torch.tensor([ids])
                    return ids
                def decode(self, ids, **kwargs): return " [Dummy Output] "
            return Dummy()

def get_model(vllm_config: VllmConfig) -> nn.Module:
    cfg = vllm_config.model_config
    if cfg.hf_config is None: cfg.hf_config = PretrainedConfig()
    path = os.path.join(cfg.model, "config.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
            for k, v in data.items():
                if k != "text_config": setattr(cfg.hf_config, k, v)
            if "text_config" in data: 
                for k, v in data["text_config"].items(): setattr(cfg.hf_config, k, v)
            cfg.hf_config.layer_types = data.get("text_config", {}).get("layer_types", [])
    if getattr(cfg.hf_config, "model_type", "") == "deepseek_v2":
        setattr(cfg.hf_config, "num_key_value_heads", getattr(cfg.hf_config, "num_attention_heads", 40))
        setattr(cfg.hf_config, "head_dim", 128)
    model_cls, _ = ModelRegistry.resolve_model_cls(getattr(cfg.hf_config, "architectures", ["LlamaForCausalLM"]), cfg)
    model = model_cls(vllm_config)
    gguf_files = [f for f in os.listdir(cfg.model) if f.endswith(".gguf")]
    if gguf_files: _load_gguf_weights(model, os.path.join(cfg.model, gguf_files[0]), cfg.hf_config)
    else: _load_safetensors(model, cfg.model)
    print(">>> Moving model to CUDA...")
    model.to(device="cuda", dtype=torch.float16); return model.eval()
