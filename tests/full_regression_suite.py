# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import time
import sys
from vllm.model_executor.models.llama import LlamaDecoderLayer
from vllm.model_executor.models.qwen3_5 import Qwen2DecoderLayer
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.quantization.tensor import AWQWeight

def run_final_audit():
    print("--- LitevLLM Final Production Audit ---")
    device = "cuda"; dtype = torch.float16
    
    def _inj(layer, q_cfg):
        for m in layer.modules():
            if "LiteLinear" in str(type(m)):
                N, K = m.output_size, m.input_size
                if q_cfg:
                    qw = torch.randint(1, 100, (N, K // 8), device=device, dtype=torch.int32)
                    sc = torch.ones((N, K // 128), device=device, dtype=dtype)
                    qz = torch.zeros((N, K // 128 // 8 + 1), device=device, dtype=torch.int32)
                    m._quant_weight = AWQWeight(qw, sc, qz, group_size=128)
                m.weight = nn.Parameter(torch.randn((N, K), device=device, dtype=dtype) * 0.01, requires_grad=False)

    def test_unit(name, layer_cls, h, i, n_h, layers, q_cfg, bs):
        print(f"\n[AUDIT] {name} | BS: {bs}")
        class Config:
            def __init__(self):
                self.hidden_size = h; self.intermediate_size = i
                self.num_attention_heads = n_h; self.num_key_value_heads = 4
                self.rms_norm_eps = 1e-6; self.max_position_embeddings = 2048; self.rope_theta = 10000.0
        
        with torch.device(device): layer = layer_cls(Config(), q_cfg, "test")
        layer = layer.half(); _inj(layer, q_cfg); layer.eval(); layer.compile_fast_dispatch()
        
        x = torch.randn((bs, 1, h), device=device, dtype=dtype)
        pos = torch.zeros((bs, 1), dtype=torch.long, device=device)
        meta = type('meta', (), {'slot_mapping': torch.zeros(bs, device=device, dtype=torch.int32),
                                 'block_tables': torch.zeros((bs, 1), device=device, dtype=torch.int32),
                                 'seq_lens': torch.ones(bs, device=device, dtype=torch.int32)})()
        kv = (torch.zeros((1, 16, 4, 128), device=device, dtype=dtype),
              torch.zeros((1, 16, 4, 128), device=device, dtype=dtype))

        # Measurement
        for _ in range(5): _ = layer(x, pos, kv, meta)
        torch.cuda.synchronize(); start = time.perf_counter()
        iters = 20
        for _ in range(iters): _ = layer(x, pos, kv, meta)
        torch.cuda.synchronize()
        lat = (time.perf_counter() - start) / iters * 1000
        tps = bs / ((lat * layers) / 1000)
        print(f"  🏁 Latency: {lat:.2f}ms | TPS: {tps:.2f}")

    # 1. Show Off Speed (TinyLlama BS=16)
    test_unit("TinyLlama (Performance)", LlamaDecoderLayer, 2048, 5632, 32, 22, None, 16)
    
    # 2. Show Stability (Qwen-9B BS=1)
    test_unit("Qwen3.5-9B (Stability)", Qwen2DecoderLayer, 3584, 18944, 28, 28, AWQConfig(), 1)

if __name__ == "__main__":
    run_final_audit()
