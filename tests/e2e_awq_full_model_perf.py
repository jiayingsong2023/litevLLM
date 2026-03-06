# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import time
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.layers.activation import Silu

class MockAWQLlamaLayer(nn.Module):
    def __init__(self, config, quant_config):
        super().__init__()
        hidden_size = config['hidden_size']
        inter_size = config['intermediate_size']
        group_size = quant_config.group_size
        
        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        
        # Attention Projections
        self.qkv_proj = LiteLinear(hidden_size, hidden_size * 3, bias=False, quant_config=quant_config)
        self.o_proj = LiteLinear(hidden_size, hidden_size, bias=False, quant_config=quant_config)
        
        # MLP
        self.gate_up_proj = LiteLinear(hidden_size, inter_size * 2, bias=False, quant_config=quant_config)
        self.down_proj = LiteLinear(inter_size, hidden_size, bias=False, quant_config=quant_config)
        self.act = Silu()
        
        # Initialize mock weights for all LiteLinear modules
        for m in self.modules():
            if isinstance(m, LiteLinear):
                # We must manually fill the qweight/qzeros/scales that init_layer set to None
                m.qweight = nn.Parameter(torch.randint(0, 100, (m.input_size, m.output_size // 8), device="cuda", dtype=torch.int32), requires_grad=False)
                m.qzeros = nn.Parameter(torch.randint(0, 100, (m.input_size // group_size, m.output_size // 8), device="cuda", dtype=torch.int32), requires_grad=False)
                m.scales = nn.Parameter(torch.randn((m.input_size // group_size, m.output_size), device="cuda", dtype=torch.float16), requires_grad=False)
                m.bias = None

    def forward(self, x):
        residual = x
        x = self.input_layernorm(x)
        x = self.qkv_proj(x)
        x = x[..., :residual.shape[-1]] 
        x = self.o_proj(x)
        x = residual + x
        
        residual = x
        x = self.post_attention_layernorm(x)
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        x = self.down_proj(self.act(gate) * up)
        x = residual + x
        return x

class MockAWQLlamaModel(nn.Module):
    def __init__(self, num_layers=32):
        super().__init__()
        self.config = {'hidden_size': 4096, 'intermediate_size': 11008}
        self.quant_config = AWQConfig(weight_bits=4, group_size=128)
        
        self.embed = nn.Embedding(32000, 4096).cuda().half()
        self.layers = nn.ModuleList([
            MockAWQLlamaLayer(self.config, self.quant_config) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(4096)
        self.head = LiteLinear(4096, 32000, bias=False, quant_config=self.quant_config)
        # Init head weights
        self.head.qweight = nn.Parameter(torch.randint(0, 100, (4096, 32000 // 8), device="cuda", dtype=torch.int32), requires_grad=False)
        self.head.qzeros = nn.Parameter(torch.randint(0, 100, (4096 // 128, 32000 // 8), device="cuda", dtype=torch.int32), requires_grad=False)
        self.head.scales = nn.Parameter(torch.randn((4096 // 128, 32000), device="cuda", dtype=torch.float16), requires_grad=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)

def run_e2e_awq_test():
    print("=== STARTING END-TO-END AWQ TRITON STRESS TEST ===")
    print("Model: Llama-7B Style (32 Layers), Quant: AWQ 4-bit")
    
    with torch.device("cuda"):
        model = MockAWQLlamaModel(num_layers=32).half()
    
    for bs in [1, 8, 32]:
        input_ids = torch.randint(0, 32000, (bs, 1), device="cuda")
        
        # Warmup
        print(f"Warmup BS={bs}...")
        for _ in range(3):
            with torch.inference_mode():
                _ = model(input_ids)
        torch.cuda.synchronize()
        
        # Benchmark
        iters = 10
        start = time.time()
        for _ in range(iters):
            with torch.inference_mode():
                _ = model(input_ids)
        torch.cuda.synchronize()
        end = time.time()
        
        latency = (end - start) / iters * 1000
        tps = (1000 / latency) * bs
        print(f"RESULT BS={bs:2}: Latency={latency:7.2f}ms, E2E Throughput={tps:8.2f} tokens/sec")

if __name__ == "__main__":
    run_e2e_awq_test()
