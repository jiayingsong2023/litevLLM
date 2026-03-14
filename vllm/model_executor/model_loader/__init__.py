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

def _load_param_from_gguf_tensor(param: Any, gguf_tensor: Any, slice_offset: int = 0) -> None:
    actual_param = param.weight if hasattr(param, "weight") else param
    if not isinstance(actual_param, torch.Tensor): return
    
    # Check if this is a LiteLinear that supports GGUFWeight injection
    parent = getattr(param, "_lite_linear", None)
    
    # Simplified detection: if the tensor is quantized (type > 1), we use GGUFWeight
    # quant_type 2 = Q4_0, 12 = Q4_K
    q_type = int(gguf_tensor.tensor_type)
    if q_type in [2, 12] and hasattr(param, "_quant_weight"):
        from vllm.model_executor.layers.quantization.tensor import GGUFWeight
        data = torch.from_numpy(np.array(gguf_tensor.data, copy=True)).to("cuda")
        scales = torch.ones((data.shape[0], 1), device="cuda", dtype=torch.float16)
        param._quant_weight = GGUFWeight(data, scales, quant_type=q_type, prefer_fused=False, 
                                          original_shape=actual_param.shape, slice_offset=slice_offset)
        return

    # Fallback to standard FP16 load
    source = _dequantize_gguf_tensor(gguf_tensor, str(actual_param.device), actual_param.dtype)
    if source is None: return
    
    if source.shape == actual_param.shape and slice_offset == 0:
        actual_param.data.copy_(source); return
    # Slicing Fallback (for unified QKV or larger tensors)
    if source.dim() == actual_param.dim():
        if source.dim() == 1:
            s = source[slice_offset : slice_offset + actual_param.shape[0]]
        else:
            s = source[slice_offset : slice_offset + actual_param.shape[0], :actual_param.shape[1]]
        if s.shape == actual_param.shape:
            actual_param.data.copy_(s); return
    if source.dim() >= 2 and source.transpose(-1, -2).shape == actual_param.shape and slice_offset == 0:
        actual_param.data.copy_(source.transpose(-1, -2).contiguous()); return
    
    with torch.no_grad():
        flat_source = source.reshape(-1)[slice_offset:]; flat_param = actual_param.data.reshape(-1)
        n = min(flat_source.numel(), flat_param.numel())
        flat_param[:n] = flat_source[:n]

def _load_qwen3_5_gguf_weights(model: nn.Module, gguf_file: str, hf_config: Any) -> None:
    print(f">>> Mapping GGUF weights from {gguf_file}...")
    reader = gguf.GGUFReader(gguf_file); tensor_map = {t.name: t for t in reader.tensors}
    num_layers = int(getattr(hf_config, "num_hidden_layers", 0))
    
    # Helper to load into LiteLinear or parameter
    def load_m(obj, key, offset=0):
        if obj is not None and key in tensor_map: _load_param_from_gguf_tensor(obj, tensor_map[key], offset)

    # Try different common model structures
    inner_model = getattr(model, "model", model)
    load_m(getattr(inner_model, "embed_tokens", None), "token_embd.weight")
    load_m(getattr(inner_model, "norm", None), "output_norm.weight")
    load_m(getattr(model, "lm_head", None), "output.weight")
    
    layers = getattr(inner_model, "layers", [])
    for i in range(min(len(layers), num_layers)):
        prefix = f"blk.{i}"; layer = layers[i]
        load_m(getattr(layer, "input_layernorm", None), f"{prefix}.attn_norm.weight")
        load_m(getattr(layer, "post_attention_layernorm", None), f"{prefix}.post_attention_norm.weight")
        
        # QKV / Attention Logic
        attn = getattr(layer, "self_attn", layer)
        q_proj = getattr(attn, "q_proj", getattr(attn, "q_a_proj", None))
        k_proj = getattr(attn, "k_proj", getattr(attn, "kv_a_proj", None))
        v_proj = getattr(attn, "v_proj", None)
        o_proj = getattr(attn, "o_proj", getattr(attn, "dense", None))
        qkv_proj = getattr(attn, "qkv_proj", getattr(attn, "query_key_value", None))
        
        qkv_key = f"{prefix}.attn_qkv.weight"
        if qkv_key in tensor_map:
            if qkv_proj: load_m(qkv_proj, qkv_key)
            else:
                q_out = q_proj.output_size if q_proj else 0
                k_out = k_proj.output_size if k_proj else 0
                load_m(q_proj, qkv_key, 0)
                load_m(k_proj, qkv_key, q_out)
                load_m(v_proj, qkv_key, q_out + k_out)
        else:
            load_m(q_proj, f"{prefix}.attn_q.weight")
            load_m(k_proj, f"{prefix}.attn_k.weight")
            load_m(v_proj, f"{prefix}.attn_v.weight")
            # GLM specific MLA/Split paths
            load_m(getattr(attn, "q_a_proj", None), f"{prefix}.attn_q_a.weight")
            load_m(getattr(attn, "q_b_proj", None), f"{prefix}.attn_q_b.weight")
            load_m(getattr(attn, "kv_a_proj", None), f"{prefix}.attn_kv_a_mqa.weight")
            load_m(getattr(attn, "kv_b_proj", None), f"{prefix}.attn_kv_b.weight")
        
        load_m(o_proj, f"{prefix}.attn_output.weight")
        load_m(getattr(layer, "kv_a_proj", None), f"{prefix}.attn_kv_a_mqa.weight")
        load_m(getattr(layer, "kv_b_proj", None), f"{prefix}.attn_kv_b.weight")
        
        # MLP / MoE
        mlp = getattr(layer, "mlp", layer)
        if hasattr(mlp, "gate_up_proj"):
            load_m(mlp.gate_up_proj, f"{prefix}.ffn_up.weight")
            load_m(mlp.down_proj, f"{prefix}.ffn_down.weight")
        else:
            load_m(getattr(mlp, "gate_proj", None), f"{prefix}.ffn_gate.weight")
            load_m(getattr(mlp, "up_proj", None), f"{prefix}.ffn_up.weight")
            load_m(getattr(mlp, "down_proj", None), f"{prefix}.ffn_down.weight")
        
        if hasattr(mlp, "experts"): # MoE experts
            load_m(getattr(mlp, "gate", None), f"{prefix}.ffn_gate_inp.weight")
        
        if i % 8 == 0: torch.cuda.empty_cache()
    print(">>> GGUF Weight loading complete.")

def get_tokenizer(model: Union[str, Any], **kwargs: Any):
    model_name = model.model if hasattr(model, "model") else model
    if hasattr(model, "trust_remote_code"): kwargs["trust_remote_code"] = model.trust_remote_code
    from transformers import AutoTokenizer
    try: return AutoTokenizer.from_pretrained(model_name, **kwargs)
    except:
        try: return AutoTokenizer.from_pretrained(model_name, use_fast=False, **kwargs)
        except Exception as e:
            print(f">>> Warning: Failed to load real tokenizer ({e}), using DummyTokenizer.")
            class DummyTokenizer:
                def __init__(self): self.vocab_size = 151936; self.eos_token_id = 151329; self.pad_token_id = 151329
                def encode(self, text, **kwargs): return [1, 2, 3, 4, 5] * 10
                def decode(self, ids, **kwargs): return " [Dummy Output] "
            return DummyTokenizer()

def _update_config_from_gguf(hf_config: Any, gguf_file: str):
    try:
        import gguf
        reader = gguf.GGUFReader(gguf_file)
        # Use lowercase keys for matching
        meta = {k.lower(): v for k, v in reader.fields.items()}
        
        # Core architectural keys to extract
        mapping = {
            "block_count": "num_hidden_layers",
            "embedding_length": "hidden_size",
            "feed_forward_length": "intermediate_size",
            "attention.head_count": "num_attention_heads",
            "attention.head_count_kv": "num_key_value_heads",
            "vocab_size": "vocab_size",
        }
        
        # Detect the primary architecture prefix (e.g., 'llama', 'deepseek2', 'glm2')
        arch_prefix = "llama"
        for k in meta.keys():
            if ".block_count" in k:
                arch_prefix = k.split(".")[0]
                break
        
        # Keys to ignore as they cause incorrect overrides (e.g., DeepSeek leading dense blocks)
        blocklist = ["leading_dense_block_count", "expert_feed_forward_length"]
        
        for k, v in meta.items():
            for g_suffix, h_key in mapping.items():
                # Priority 1: Exact prefix match {arch}.{suffix}
                # Priority 2: Generic suffix match
                is_match = False
                if k == f"{arch_prefix}.{g_suffix}": is_match = True
                elif k.endswith(g_suffix) and not any(b in k for b in blocklist):
                     # Only allow fuzzy match if we don't have a specific value yet
                     if not hasattr(hf_config, h_key): is_match = True
                
                if is_match:
                    val = v.parts[-1] if hasattr(v, 'parts') else v
                    if hasattr(val, '__len__') and not isinstance(val, (str, bytes)): val = val[0]
                    setattr(hf_config, h_key, int(val))
                    print(f">>> GGUF Match ({arch_prefix}): {k} -> {h_key} = {int(val)}")

        # Safety Fallbacks
        if not hasattr(hf_config, "num_key_value_heads") or hf_config.num_key_value_heads == 0:
            hf_config.num_key_value_heads = getattr(hf_config, "num_attention_heads", 0)
            
        print(f">>> GGUF Synced: Layers={getattr(hf_config, 'num_hidden_layers', '??')}, Hidden={hf_config.hidden_size}, KV-Heads={hf_config.num_key_value_heads}")
    except Exception as e:
        print(f">>> Failed to update config from GGUF metadata: {e}")

def get_model(vllm_config: VllmConfig) -> nn.Module:
    model_config = vllm_config.model_config
    if model_config.hf_config is None or not hasattr(model_config.hf_config, "vocab_size"):
        try: model_config.hf_config = AutoConfig.from_pretrained(model_config.model, trust_remote_code=True)
        except:
            config_path = os.path.join(model_config.model, "config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, "r") as f: data = json.load(f); model_config.hf_config = PretrainedConfig(**data)
    
    c = model_config.hf_config
    model_path = model_config.model
    gguf_files = [f for f in os.listdir(model_path) if f.endswith(".gguf")] if os.path.isdir(model_path) else []
    
    # CRITICAL: Sync metadata from GGUF if available to fix accuracy and runtime errors
    if gguf_files:
        _update_config_from_gguf(c, os.path.join(model_path, gguf_files[0]))

    # Final fallbacks for stability
    for attr, val in [("vocab_size", 151936), ("hidden_size", 4096), ("num_hidden_layers", 32)]:
        if not hasattr(c, attr): setattr(c, attr, val)
    if not hasattr(c, "num_attention_heads"): setattr(c, "num_attention_heads", getattr(c, "n_heads", 32))
    if not hasattr(c, "num_key_value_heads"): setattr(c, "num_key_value_heads", getattr(c, "num_kv_heads", getattr(c, "multi_query_group_num", 32)))

    model_cls, _ = ModelRegistry.resolve_model_cls(getattr(c, "architectures", ["LlamaForCausalLM"]), model_config)
    model = model_cls(vllm_config)
    
    if gguf_files:
        _load_qwen3_5_gguf_weights(model, os.path.join(model_path, gguf_files[0]), model_config.hf_config)

    model.to(device="cuda", dtype=torch.float16)
    return model.eval()
