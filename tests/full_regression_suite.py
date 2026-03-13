# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import time
import sys
from vllm.model_executor.models.qwen3_5 import Qwen2DecoderLayer
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.quantization.tensor import AWQWeight

def run_peak_audit_ladder():
    print("--- Qwen-9B Performance Climbing Audit (Ladder Mode) ---")
    device = "cuda"; dtype = torch.float16
    
    class Config:
        def __init__(self):
            self.hidden_size = 3584; self.intermediate_size = 18944
            self.num_attention_heads = 28; self.num_key_value_heads = 4
            self.rms_norm_eps = 1e-6; self.max_position_embeddings = 2048; self.rope_theta = 1000000.0
    
    with torch.device(device):
        layer = Qwen2DecoderLayer(Config(), AWQConfig(), "peak")
    layer = layer.half().to(device)
    
    # Precise Injection
    for m in layer.modules():
        if "LiteLinear" in str(type(m)):
            N, K = m.output_size, m.input_size
            qw = torch.randint(1, 100, (N, K // 8), device=device, dtype=torch.int32)
            sc = torch.ones((N, K // 128), device=device, dtype=dtype)
            qz = torch.zeros((N, K // 128 // 8 + 1), device=device, dtype=torch.int32)
            m._quant_weight = AWQWeight(qw, sc, qz, group_size=128)
            m.weight = nn.Parameter(torch.randn((N, K), device=device, dtype=dtype) * 0.01, requires_grad=False)

    print("Compiling Slice-Mode Dispatch...")
    layer.eval(); layer.compile_fast_dispatch()
    
    ladder = [8]
    for bs in ladder:
        print(f"\nClimbing to BS={bs}...")
        try:
            x = torch.randn((bs, 1, 3584), device=device, dtype=dtype)
            pos = torch.zeros((bs, 1), dtype=torch.long, device=device)
            meta = type('meta', (), {'slot_mapping': torch.zeros(bs, device=device, dtype=torch.int32),
                                     'block_tables': torch.zeros((bs, 1), device=device, dtype=torch.int32),
                                     'seq_lens': torch.ones(bs, device=device, dtype=torch.int32)})()
            kv = (torch.zeros((1, 16, 4, 128), device=device, dtype=dtype),
                  torch.zeros((1, 16, 4, 128), device=device, dtype=dtype))

            with torch.inference_mode():
                for _ in range(1): _ = layer(x, pos, kv, meta)
            torch.cuda.synchronize()
            
            iters = 5; start = time.perf_counter()
            with torch.inference_mode():
                for _ in range(iters): _ = layer(x, pos, kv, meta)
            torch.cuda.synchronize()
            
            lat = (time.perf_counter() - start) / iters * 1000
            tps_est = bs / ((lat * 28) / 1000)
            print(f"  🏆 REACHED! Latency: {lat:.2f}ms | Est TPS (28L): {tps_est:.2f}")
        except Exception as e:
            print(f"  ❌ FAILED at BS={bs}: {e}")
            break # Stop climbing if we hit hardware limit

if __name__ == "__main__":
    run_peak_audit_ladder()
