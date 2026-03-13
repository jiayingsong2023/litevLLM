# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
from vllm.model_executor.models.llama import LlamaDecoderLayer
from vllm.model_executor.models.qwen3_5 import Qwen2DecoderLayer
from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.quantization.gguf import GGUFConfig

class LayerWiseRegressionSuite:
    def __init__(self):
        self.device = "cuda"; self.dtype = torch.float16
        self.results = []

    def inject_mock(self, layer):
        from vllm.model_executor.layers.lite_linear import LiteLinear
        for m in layer.modules():
            if isinstance(m, LiteLinear):
                N, K = m.output_size, m.input_size
                m.qweight = nn.Parameter(torch.randint(1, 100, (N, K // 8), device=self.device, dtype=torch.int32), requires_grad=False)
                m.scales = nn.Parameter(torch.ones((N, K // 128), device=self.device, dtype=self.dtype), requires_grad=False)
                m.qzeros = nn.Parameter(torch.zeros((N, K // 128 // 8 + 1), device=self.device, dtype=torch.int32), requires_grad=False)
                m.weight = nn.Parameter(torch.randn((N, K), device=self.device, dtype=self.dtype) * 0.01, requires_grad=False)

    def run_unit(self, name, layer_cls, h, i, n_h, n_kv, layers, q_cfg, bs=16):
        print(f"\n[REGRESSION] {name} | BS: {bs} | Layers: {layers}")
        try:
            class Config:
                def __init__(self):
                    self.hidden_size = h; self.intermediate_size = i
                    self.num_attention_heads = n_h; self.num_key_value_heads = n_kv
                    self.rms_norm_eps = 1e-6; self.max_position_embeddings = 2048; self.rope_theta = 10000.0
            
            with torch.device(self.device):
                layer = layer_cls(Config(), q_cfg, "test")
            layer.half().eval(); self.inject_mock(layer)
            layer.compile_fast_dispatch()
            
            x = torch.randn((bs, 1, h), device=self.device, dtype=self.dtype)
            pos = torch.zeros((bs, 1), dtype=torch.long, device=self.device)
            meta = type('meta', (), {'slot_mapping': torch.zeros(bs, device=self.device, dtype=torch.int32),
                                     'block_tables': torch.zeros((bs, 1), device=self.device, dtype=torch.int32),
                                     'seq_lens': torch.ones(bs, device=self.device, dtype=torch.int32)})()
            kv = (torch.zeros((1, 16, n_kv, h//n_h), device=self.device, dtype=self.dtype),
                  torch.zeros((1, 16, n_kv, h//n_h), device=self.device, dtype=self.dtype))

            for _ in range(5): _ = layer(x, pos, kv, meta)
            torch.cuda.synchronize()
            
            iters = 50; start = time.perf_counter()
            for _ in range(iters): _ = layer(x, pos, kv, meta)
            torch.cuda.synchronize()
            
            lat_ms = ((time.perf_counter() - start) / iters) * 1000
            tps = bs / ((lat_ms * layers) / 1000)
            
            print(f"  🏁 RESULT: {tps:.2f} t/s (E2E Est)")
            self.results.append({"name": name, "tps": tps, "status": "✅ PASS"})
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            self.results.append({"name": name, "tps": 0.0, "status": "❌ FAIL"})

if __name__ == "__main__":
    runner = LayerWiseRegressionSuite()
    # Batch Size 16 is the Sweet Spot for gfx1151 Stability
    runner.run_unit("TinyLlama", LlamaDecoderLayer, 2048, 5632, 32, 4, 22, None, 16)
    runner.run_unit("Qwen3.5-9B", Qwen2DecoderLayer, 3584, 18944, 28, 4, 28, AWQConfig(), 16)
    runner.run_unit("DeepSeek-Lite", DeepseekV2DecoderLayer, 2048, 12288, 16, 16, 27, GGUFConfig(), 16)
    
    print("\n" + "="*60 + "\nSTABLE PERFORMANCE SUMMARY (ESTIMATED)\n" + "-"*60)
    for r in runner.results:
        print(f"{r['name']:<20} | {r['tps']:<10.2f} | {r['status']}")
    print("="*60)
    if any(r['status'] == "❌ FAIL" for r in runner.results): sys.exit(1)
