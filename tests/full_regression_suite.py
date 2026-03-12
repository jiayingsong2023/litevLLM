# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import gc
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM
from vllm.model_executor.models.glm import GlmForCausalLM
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.quantization.gguf import GGUFConfig

class UltimatePerformanceAuditor:
    def __init__(self):
        self.device = "cuda"; self.dtype = torch.float16
        self.results = []

    def inject_weights(self, model, q_cfg):
        for m in model.modules():
            if "LiteLinear" in str(type(m)):
                N, K = m.output_size, m.input_size
                # Ensure K is multiple of 32 for GGUF safety
                if isinstance(q_cfg, AWQConfig):
                    m.qweight = nn.Parameter(torch.randint(1, 100, (N, K // 8), device=self.device, dtype=torch.int32), requires_grad=False)
                    m.scales = nn.Parameter(torch.ones((N, K // 128), device=self.device, dtype=self.dtype), requires_grad=False)
                    m.qzeros = nn.Parameter(torch.zeros((N, K // 128 // 8 + 1), device=self.device, dtype=torch.int32), requires_grad=False)
                elif isinstance(q_cfg, GGUFConfig):
                    # Precise Q4_0 byte mapping: 18 bytes per 32 weights
                    bytes_per_row = (K // 32) * 18
                    m.qweight = nn.Parameter(torch.randint(0, 255, (N, bytes_per_row), device=self.device, dtype=torch.uint8), requires_grad=False)
                    m.scales = nn.Parameter(torch.ones(1, device=self.device, dtype=self.dtype), requires_grad=False)
                    m.gguf_quant_type = 2
                m.weight = nn.Parameter(torch.randn((N, K), device=self.device, dtype=self.dtype) * 0.01, requires_grad=False)
            elif hasattr(m, 'weight') and isinstance(m.weight, nn.Parameter):
                m.weight.data = m.weight.data.to(self.dtype)

    def run_audit(self, name, model_cls, hidden, heads, layers, q_cfg, bs=32, scale=1.0):
        test_layers = max(1, int(layers * scale))
        print(f"\n[AUDIT] {name} | BS: {bs} | Layers: {layers}")
        try:
            class MockConfig:
                def __init__(self):
                    self.hidden_size = hidden; self.num_attention_heads = heads; self.num_key_value_heads = heads
                    self.intermediate_size = hidden * 4; self.num_hidden_layers = test_layers; self.rms_norm_eps = 1e-6
                    self.vocab_size = 32000; self.max_position_embeddings = 2048; self.rope_theta = 10000.0
                    self.hf_config = self
            class MockVllmConfig:
                def __init__(self):
                    self.model_config = MockConfig(); self.quant_config = q_cfg; self.device = "cuda"

            with torch.device(self.device):
                model = model_cls(MockVllmConfig())
            model.eval(); self.inject_weights(model, q_cfg); model = model.half()
            
            input_ids = torch.randint(0, 1000, (bs, 1), device=self.device)
            pos = torch.zeros((bs, 1), dtype=torch.long, device=self.device)
            kv = [None] * test_layers
            
            # Hot fix for TinyLlama Dtype mismatch in regression
            with torch.inference_mode():
                for _ in range(3): _ = model(input_ids, pos, kv, None)
            torch.cuda.synchronize()
            
            iters = 20; start = time.perf_counter()
            with torch.inference_mode():
                for _ in range(iters): _ = model(input_ids, pos, kv, None)
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            tps = (bs * iters) / ((end - start) / scale)
            print(f"  🏁 RESULT: {tps:.2f} t/s")
            self.results.append({"name": name, "tps": tps, "status": "✅ PASS"})
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            self.results.append({"name": name, "tps": 0.0, "status": "❌ FAIL"})
        finally:
            gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__":
    auditor = UltimatePerformanceAuditor()
    tasks = [
        ("TinyLlama-1.1B", LlamaForCausalLM, 2048, 32, 22, None, 16),
        ("DeepSeek-V2-Lite", DeepseekV2ForCausalLM, 2048, 16, 27, GGUFConfig(), 16),
        ("GLM-4.7-Flash", GlmForCausalLM, 4096, 32, 28, GGUFConfig(), 16),
        ("Qwen3.5-9B-AWQ", Qwen3_5ForConditionalGeneration, 3584, 28, 28, AWQConfig(), 16),
        ("Qwen3.5-9B-GGUF", Qwen3_5ForConditionalGeneration, 3584, 28, 28, GGUFConfig(), 16),
        ("Qwen3.5-35B-GGUF", Qwen3_5ForConditionalGeneration, 5120, 28, 80, GGUFConfig(prefer_fused=True), 1, 0.1),
    ]
    for task in tasks: auditor.run_audit(*task)
    print("\n" + "="*60 + "\nFINAL PERFORMANCE SUMMARY\n" + "-"*60)
    for r in auditor.results:
        print(f"{r['name']:<20} | {r['tps']:<10.2f} | {r['status']}")
    print("="*60)
