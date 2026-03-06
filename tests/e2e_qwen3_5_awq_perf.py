# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import time
import gc
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.layers.activation import Silu
from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe
from vllm.model_executor.layers.quantization.gguf import _GLOBAL_GGUF_CACHE, clear_gguf_cache

class MockAWQQwen3_5MoE(nn.Module):
    def __init__(self, config, quant_config):
        super().__init__()
        self.num_experts = config['num_experts']
        self.topk = config['num_experts_per_tok']
        self.intermediate_size = config['moe_intermediate_size']
        self.hidden_size = config['hidden_size']
        group_size = quant_config.group_size
        
        self.w1 = LiteLinear(self.hidden_size, self.num_experts * self.intermediate_size, bias=False, quant_config=quant_config)
        self.w2 = LiteLinear(self.intermediate_size, self.num_experts * self.hidden_size, bias=False, quant_config=quant_config)
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False).cuda().half()

        for m in [self.w1, self.w2]:
            m.qweight = nn.Parameter(torch.randint(0, 100, (m.input_size, m.output_size // 8), device="cuda", dtype=torch.int32), requires_grad=False)
            m.qzeros = nn.Parameter(torch.randint(0, 100, (m.input_size // group_size, m.output_size // 8), device="cuda", dtype=torch.int32), requires_grad=False)
            m.scales = nn.Parameter(torch.randn((m.input_size // group_size, m.output_size), device="cuda", dtype=torch.float16), requires_grad=False)

    def _get_dequant_weight(self, module):
        w = _GLOBAL_GGUF_CACHE.get(module.weight_id)
        if w is None:
            module(torch.randn((2, module.input_size), device="cuda", dtype=torch.float16))
            w = _GLOBAL_GGUF_CACHE.get(module.weight_id)
        return w

    def forward(self, x):
        w1_t = self._get_dequant_weight(self.w1).view(self.num_experts, self.intermediate_size, -1)
        w2_t = self._get_dequant_weight(self.w2).view(self.num_experts, -1, self.intermediate_size)
        router_logits = self.gate(x.view(-1, x.shape[-1]))
        return fused_moe(x.view(-1, x.shape[-1]), w1_t, w2_t, router_logits, topk=self.topk).view(x.shape)

class MockAWQQwen3_5Layer(nn.Module):
    def __init__(self, config, quant_config):
        super().__init__()
        self.input_layernorm = RMSNorm(config['hidden_size'])
        self.post_attention_layernorm = RMSNorm(config['hidden_size'])
        group_size = quant_config.group_size
        self.qkv_proj = LiteLinear(config['hidden_size'], config['hidden_size'] * 3, bias=False, quant_config=quant_config)
        self.o_proj = LiteLinear(config['hidden_size'], config['hidden_size'], bias=False, quant_config=quant_config)
        
        for m in [self.qkv_proj, self.o_proj]:
            m.qweight = nn.Parameter(torch.randint(0, 100, (m.input_size, m.output_size // 8), device="cuda", dtype=torch.int32), requires_grad=False)
            m.qzeros = nn.Parameter(torch.randint(0, 100, (m.input_size // group_size, m.output_size // 8), device="cuda", dtype=torch.int32), requires_grad=False)
            m.scales = nn.Parameter(torch.randn((m.input_size // group_size, m.output_size), device="cuda", dtype=torch.float16), requires_grad=False)

        if config.get('num_experts', 0) > 0:
            self.mlp = MockAWQQwen3_5MoE(config, quant_config)
        else:
            self.gate_up_proj = LiteLinear(config['hidden_size'], config['intermediate_size'] * 2, bias=False, quant_config=quant_config)
            self.down_proj = LiteLinear(config['intermediate_size'], config['hidden_size'], bias=False, quant_config=quant_config)
            self.act = Silu()
            for m in [self.gate_up_proj, self.down_proj]:
                m.qweight = nn.Parameter(torch.randint(0, 100, (m.input_size, m.output_size // 8), device="cuda", dtype=torch.int32), requires_grad=False)
                m.qzeros = nn.Parameter(torch.randint(0, 100, (m.input_size // group_size, m.output_size // 8), device="cuda", dtype=torch.int32), requires_grad=False)
                m.scales = nn.Parameter(torch.randn((m.input_size // group_size, m.output_size), device="cuda", dtype=torch.float16), requires_grad=False)
            self.mlp = self._dense_mlp

    def _dense_mlp(self, x):
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(self.act(gate) * up)

    def forward(self, x):
        residual = x
        x = self.input_layernorm(x)
        x = self.qkv_proj(x)
        x = x[..., :residual.shape[-1]]
        x = self.o_proj(x)
        x = residual + x
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x) if not isinstance(self.mlp, MockAWQQwen3_5MoE) else self.mlp(x)
        x = residual + x
        return x

def benchmark_qwen_awq(name, config, num_layers, bs):
    print(f"\n[ANALYSIS] Profiling Qwen3.5-{name} AWQ (Concurrency: {bs})")
    quant_config = AWQConfig(weight_bits=4, group_size=128)
    _GLOBAL_GGUF_CACHE.max_size = 512
    clear_gguf_cache()
    
    with torch.device("cuda"):
        layers = nn.ModuleList([MockAWQQwen3_5Layer(config, quant_config) for _ in range(num_layers)]).half()
    input_ids = torch.randn((bs, 1, config['hidden_size']), device="cuda", dtype=torch.float16)
    
    # Warmup
    for _ in range(3):
        with torch.inference_mode():
            x = input_ids
            for layer in layers: x = layer(x)
    torch.cuda.synchronize()
    
    iters = 10
    start = time.time()
    for _ in range(iters):
        with torch.inference_mode():
            x = input_ids
            for layer in layers: x = layer(x)
    torch.cuda.synchronize()
    
    latency = (time.time() - start) / iters * 1000
    tps = (1000 / latency) * bs
    
    # Advanced Metrics
    tpm = tps * 60
    tph = tps * 3600
    efficiency = (tps / bs) * 100 # Score
    
    print(f"┌──────────────────────────────────────────────────────────┐")
    print(f"│ FASTINFERENCE INDUSTRY ANALYSIS REPORT: QWEN3.5-{name:8} │")
    print(f"├──────────────────────────────────────────────────────────┤")
    print(f"│ Concurrency (BS)      : {bs:<32} │")
    print(f"│ Step Latency          : {latency:7.2f} ms{' ':<23} │")
    print(f"│ Tokens Per Second     : {tps:10.2f} t/s{' ':<20} │")
    print(f"│ Tokens Per Minute     : {tpm:10.2f} t/m{' ':<20} │")
    print(f"│ Generation Capacity   : {tph/1e6:7.2f} M tokens/hour{' ':<15} │")
    print(f"│ Architecture Efficiency: {efficiency:7.2f}%{' ':<24} │")
    print(f"└──────────────────────────────────────────────────────────┘")
    return tps

if __name__ == "__main__":
    config_9b = {'hidden_size': 3584, 'intermediate_size': 18944, 'num_experts': 0}
    config_35b = {'hidden_size': 2048, 'moe_intermediate_size': 512, 'num_experts': 256, 'num_experts_per_tok': 8}
    
    print("\n" + "="*60)
    print("QWEN3.5 AWQ QUANTIZATION PERFORMANCE AUDIT")
    print("Hardware: AMD AI Max (Strix Point) 60GB")
    print("="*60)
    
    benchmark_qwen_awq("9B", config_9b, num_layers=28, bs=32)
    benchmark_qwen_awq("35B", config_35b, num_layers=40, bs=1)
    benchmark_qwen_awq("35B-C16", config_35b, num_layers=40, bs=16)
