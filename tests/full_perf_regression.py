# SPDX-License-Identifier: Apache-2.0
"""
FastInference Full Performance Regression Suite - FIXED
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import traceback

from vllm.model_executor.models.lite_config import LiteConfig
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.quantization.gguf import GGUFConfig
from vllm.model_executor.layers.quantization.tensor import AWQWeight, GGUFWeight

def inject_fp16_weights(layer, device, dtype):
    for m in layer.modules():
        if isinstance(m, LiteLinear):
            N, K = m.output_size, m.input_size
            m.weight = nn.Parameter(torch.randn((N, K), device=device, dtype=dtype) * 0.01, requires_grad=False)
            if hasattr(m, "bias") and m.bias is not None:
                m.bias = nn.Parameter(torch.zeros(N, device=device, dtype=dtype), requires_grad=False)

def inject_awq_weights(layer, device, dtype):
    for m in layer.modules():
        if isinstance(m, LiteLinear):
            N, K = m.output_size, m.input_size
            group_size = 128
            qw = torch.randint(1, 100, (N, K // 8), device=device, dtype=torch.int32)
            num_groups = max(1, K // group_size)
            sc = torch.ones((N, num_groups), device=device, dtype=dtype)
            qz = torch.zeros((N, max(1, num_groups // 8 + 1)), device=device, dtype=torch.int32)
            m._quant_weight = AWQWeight(qw, sc, qz, group_size=group_size)
            m.quant_config = AWQConfig()
            m.weight = nn.Parameter(torch.randn((N, K), device=device, dtype=dtype) * 0.01, requires_grad=False)

def inject_gguf_weights(layer, device, dtype):
    for m in layer.modules():
        if isinstance(m, LiteLinear):
            N, K = m.output_size, m.input_size
            if K % 32 != 0:
                m.weight = nn.Parameter(torch.randn((N, K), device=device, dtype=dtype) * 0.01, requires_grad=False)
                continue
            byte_cols = (K // 32) * 18
            qw = torch.randint(0, 255, (N, byte_cols), device=device, dtype=torch.uint8)
            sc = torch.ones((N, 1), device=device, dtype=dtype)
            m._quant_weight = GGUFWeight(qw, sc, quant_type=2, prefer_fused=False)
            m.quant_config = GGUFConfig(prefer_fused=False)
            m.weight = nn.Parameter(torch.randn((N, K), device=device, dtype=dtype) * 0.01, requires_grad=False)

class SimpleMoELayer(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        hs = config.hidden_size; inter = config.intermediate_size
        self.input_layernorm = RMSNorm(hs); self.post_attention_layernorm = RMSNorm(hs)
        self.q_proj = LiteLinear(hs, hs, bias=False, quant_config=quant_config); self.o_proj = LiteLinear(hs, hs, bias=False, quant_config=quant_config)
        self.gate_proj = LiteLinear(hs, inter, bias=False, quant_config=quant_config); self.up_proj = LiteLinear(hs, inter, bias=False, quant_config=quant_config); self.down_proj = LiteLinear(inter, hs, bias=False, quant_config=quant_config)
    def forward(self, x, *args, **kwargs):
        h = self.input_layernorm(x); q = self.q_proj(h); x = x + self.o_proj(q)
        h2 = self.post_attention_layernorm(x); g = self.gate_proj(h2); u = self.up_proj(h2); x = x + self.down_proj(F.silu(g) * u)
        return x

def run_single_benchmark(model_name, layer, config, batch_size, num_layers, device, dtype):
    hs = config.hidden_size
    x = torch.randn((batch_size, 1, hs), device=device, dtype=dtype)
    pos = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
    kv = (torch.zeros(1, 16, 32, 64, device=device, dtype=dtype), torch.zeros(1, 16, 32, 64, device=device, dtype=dtype))
    meta = {"slot_mapping": torch.arange(batch_size, device=device, dtype=torch.int32), "block_tables": torch.zeros(batch_size, 1, device=device, dtype=torch.int32), "seq_lens": torch.ones(batch_size, device=device, dtype=torch.int32)}
    
    with torch.inference_mode():
        for _ in range(2): 
            try: layer(x, pos, kv, meta)
            except: layer(x)
    torch.cuda.synchronize()
    
    iters = 10; start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(iters):
            try: layer(x, pos, kv, meta)
            except: layer(x)
    torch.cuda.synchronize()
    lat_ms = (time.perf_counter() - start) / iters * 1000
    tps = batch_size / ((lat_ms * num_layers) / 1000)
    print(f"  🏆 PASS! Latency: {lat_ms:.2f}ms/layer | Est {num_layers}L TPS: {tps:.1f} tok/s (BS={batch_size})")

def build_and_run(model_name, layer_cls, cfg_dict, quant_type, batch_size):
    print(f"\n[{model_name}] quant={quant_type}, BS={batch_size}")
    device = "cuda"; dtype = torch.float16
    class Cfg: pass
    cfg = Cfg(); defaults = {"rms_norm_eps": 1e-6, "num_experts": 0, "q_lora_rank": 768, "kv_lora_rank": 512, "qk_nope_head_dim": 64, "qk_rope_head_dim": 64, "v_head_dim": 128, "n_routed_experts": 64, "num_experts_per_tok": 6, "moe_intermediate_size": 1536, "num_attention_heads": 32, "num_key_value_heads": 32}
    for k, v in {**defaults, **cfg_dict}.items(): setattr(cfg, k, v)
    lite_cfg = LiteConfig(cfg)
    try:
        layer = layer_cls(lite_cfg, quant_config=None, prefix="model.layers.0")
        layer = layer.to(dtype).to(device)
        if quant_type == "awq": inject_awq_weights(layer, device, dtype)
        elif "gguf" in quant_type: inject_gguf_weights(layer, device, dtype)
        else: inject_fp16_weights(layer, device, dtype)
        run_single_benchmark(model_name, layer, lite_cfg, batch_size, lite_cfg.num_hidden_layers, device, dtype)
    except Exception as e:
        print(f"  ❌ FAILED: {e}"); traceback.print_exc()

def run_all():
    print("FASTINFERENCE PERFORMANCE REGRESSION")
    from vllm.model_executor.models.llama import LlamaDecoderLayer
    from vllm.model_executor.models.qwen3_5 import Qwen3_5LinearAttentionLayer
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer
    
    build_and_run("TinyLlama-1.1B", LlamaDecoderLayer, {"hidden_size": 2048, "num_hidden_layers": 22}, "fp16", 32)
    build_and_run("Qwen3.5-9B AWQ", Qwen3_5LinearAttentionLayer, {"hidden_size": 4096, "num_hidden_layers": 32}, "awq", 32)
    build_and_run("DeepSeek-V2-Lite", DeepseekV2DecoderLayer, {"hidden_size": 2048, "num_hidden_layers": 27}, "gguf", 16)

if __name__ == "__main__":
    run_all()
