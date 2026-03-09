# SPDX-License-Identifier: Apache-2.0
import os
import torch
import torch.nn as nn
from typing import Any, List, Optional, Union
import numpy as np
import gguf
import json
from transformers import AutoTokenizer, AutoConfig, PretrainedConfig
from vllm.config import VllmConfig, ModelConfig
from vllm.model_executor.models.registry import ModelRegistry

def _dequantize_gguf_tensor(gguf_tensor: Any, device: str, dtype: torch.dtype) -> torch.Tensor:
    return torch.from_numpy(np.array(gguf_tensor.data, copy=True)).to(device=device).to(dtype=dtype)

def _load_param_from_gguf_tensor(param: Any, gguf_tensor: Any) -> None:
    if param is None or gguf_tensor is None: return
    actual_param = param.weight if hasattr(param, "weight") else param
    if actual_param is None: return
    source = _dequantize_gguf_tensor(gguf_tensor, str(actual_param.device), actual_param.dtype)
    if source is None: return
    if source.shape == actual_param.shape:
        actual_param.data.copy_(source); return
    with torch.no_grad():
        flat_source = source.reshape(-1); flat_param = actual_param.data.reshape(-1)
        n = min(flat_source.numel(), flat_param.numel())
        flat_param[:n] = flat_source[:n]

def _load_qwen3_5_gguf_weights(model: nn.Module, gguf_file: str, hf_config: Any) -> None:
    reader = gguf.GGUFReader(gguf_file); tensor_map = {t.name: t for t in reader.tensors}
    num_layers = int(getattr(hf_config, "num_hidden_layers", 0))
    _load_param_from_gguf_tensor(model.model.embed_tokens, tensor_map.get("token_embd.weight"))
    _load_param_from_gguf_tensor(model.model.norm, tensor_map.get("output_norm.weight"))
    _load_param_from_gguf_tensor(model.lm_head, tensor_map.get("output.weight"))
    for i in range(num_layers):
        prefix = f"blk.{i}"; layer = model.model.layers[i]
        _load_param_from_gguf_tensor(layer.input_layernorm, tensor_map.get(f"{prefix}.attn_norm.weight"))
        _load_param_from_gguf_tensor(layer.post_attention_layernorm, tensor_map.get(f"{prefix}.post_attention_norm.weight"))
        qkv_obj = None
        for attr in ["qkv_proj", "kv_proj", "attn_qkv", "attn_q"]:
            if hasattr(layer.self_attn, attr): qkv_obj = getattr(layer.self_attn, attr); break
        for k in [f"{prefix}.attn_qkv.weight", f"{prefix}.self_attn.qkv_proj.weight", f"{prefix}.attn_q.weight", f"{prefix}.attn_kv_a_mqa.weight"]:
            if k in tensor_map: _load_param_from_gguf_tensor(qkv_obj, tensor_map[k]); break
        _load_param_from_gguf_tensor(layer.self_attn.o_proj, tensor_map.get(f"{prefix}.attn_output.weight"))
        if hasattr(layer.mlp, "gate_up_proj"):
            for k in [f"{prefix}.ffn_up.weight", f"{prefix}.ffn_gate.weight"]:
                if k in tensor_map: _load_param_from_gguf_tensor(layer.mlp.gate_up_proj, tensor_map[k]); break
            _load_param_from_gguf_tensor(layer.mlp.down_proj, tensor_map.get(f"{prefix}.ffn_down.weight"))
        if i % 8 == 0: torch.cuda.empty_cache()

def get_tokenizer(model_name: str, **kwargs: Any): return AutoTokenizer.from_pretrained(model_name, **kwargs)

def get_model(vllm_config: VllmConfig) -> nn.Module:
    model_config = vllm_config.model_config
    if model_config.hf_config is None or not hasattr(model_config.hf_config, "vocab_size"):
        try: model_config.hf_config = AutoConfig.from_pretrained(model_config.model, trust_remote_code=True)
        except:
            config_path = os.path.join(model_config.model, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f: data = json.load(f); model_config.hf_config = PretrainedConfig(**data)
    c = model_config.hf_config
    for attr, val in [("vocab_size", 151936), ("hidden_size", 4096), ("num_hidden_layers", 32)]:
        if not hasattr(c, attr): setattr(c, attr, val)
    if not hasattr(c, "num_attention_heads"): setattr(c, "num_attention_heads", getattr(c, "n_heads", 32))
    if not hasattr(c, "num_key_value_heads"): setattr(c, "num_key_value_heads", getattr(c, "num_kv_heads", getattr(c, "multi_query_group_num", 32)))

    model_cls, _ = ModelRegistry.resolve_model_cls(getattr(c, "architectures", ["LlamaForCausalLM"]), model_config)
    # CREATE ON CUDA, BUT INDIVIDUAL LAYERS WILL SELF-OFFLOAD TO CPU IN __INIT__
    model = model_cls(vllm_config)
    model_path = model_config.model
    if os.path.isdir(model_path):
        gguf_files = [f for f in os.listdir(model_path) if f.endswith(".gguf")]
        if gguf_files: _load_qwen3_5_gguf_weights(model, os.path.join(model_path, gguf_files[0]), model_config.hf_config)
    return model.eval()
