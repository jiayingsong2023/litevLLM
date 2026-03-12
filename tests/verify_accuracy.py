# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn.functional as F
import numpy as np
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.kernels.triton.rmsnorm_awq_fused import rmsnorm_awq_fused_linear
from vllm.model_executor.layers.quantization.awq_triton import awq_dequantize_triton

def verify_awq_macro_fusion():
    device = "cuda"
    dtype = torch.float16
    bs, hidden_size, out_size = 32, 3584, 18944 # Qwen-9B MLP Gate/Up size
    group_size = 128
    eps = 1e-6

    print(f"--- Verifying AWQ Macro-Fusion Accuracy ({bs}x{hidden_size}x{out_size}) ---")

    # 1. Prepare Data
    x = torch.randn((bs, hidden_size), device=device, dtype=dtype)
    norm_w = torch.randn((hidden_size,), device=device, dtype=dtype)
    
    qweight = torch.randint(0, 100, (out_size, hidden_size // 8), device=device, dtype=torch.int32)
    scales = torch.randn((out_size, hidden_size // group_size), device=device, dtype=dtype)
    qzeros = torch.randint(0, 100, (out_size, hidden_size // group_size // 8 + 1), device=device, dtype=torch.int32)

    # 2. Reference Path (Standard PyTorch)
    # Step A: RMSNorm
    ref_norm = (x.to(torch.float32) * torch.rsqrt(x.to(torch.float32).pow(2).mean(-1, keepdim=True) + eps)) * norm_w.to(torch.float32)
    # Step B: Dequantize Weight
    w_fp16 = awq_dequantize_triton(qweight, scales, qzeros, group_size)
    print(f"DEBUG: Reference Dequant Weight Shape: {w_fp16.shape}")
    print(f"DEBUG: Reference Dequant Weight Stride: {w_fp16.stride()}")
    # Step C: Matmul
    ref_out = F.linear(ref_norm.to(dtype), w_fp16)
    print(f"DEBUG: Reference Output Mean: {ref_out.mean().item():.4f}")

    # 3. Optimized Path (Macro-Fused Triton Kernel)
    opt_out = rmsnorm_awq_fused_linear(x, qweight, scales, qzeros, group_size, norm_w, eps)
    print(f"DEBUG: Optimized Output Mean: {opt_out.mean().item():.4f}")

    # 4. Metrics
    # Ensure layouts are same for comparison
    ref_flat = ref_out.to(torch.float32).flatten()
    opt_flat = opt_out.to(torch.float32).flatten()
    
    cos_sim = F.cosine_similarity(ref_flat, opt_flat, dim=0).item()
    max_diff = (ref_out - opt_out).abs().max().item()
    mean_diff = (ref_out - opt_out).abs().mean().item()

    print(f"Cosine Similarity: {cos_sim:.8f}")
    print(f"Max Absolute Diff: {max_diff:.8f}")
    print(f"Mean Absolute Diff: {mean_diff:.8f}")

    # Thresholds adjusted for FP16 large-scale accumulation effects
    if cos_sim > 0.9999 and max_diff < 2.0:
        print("✅ PASS: Optimized kernel matches reference implementation (within FP16 rounding limits).")
        return True
    else:
        print("❌ FAIL: Significant precision loss detected!")
        return False

def verify_fast_dispatch_logic():
    # This verifies if the pre-binding logic in llama.py correctly passes data
    # without dropping weights or applying wrong parameters.
    from vllm.model_executor.models.llama import LlamaModel
    from vllm.model_executor.layers.quantization.awq import AWQConfig
    
    device = "cuda"
    print("\n--- Verifying Fast Dispatch Logic (Layer-level) ---")
    
    class DummyConfig:
        def __init__(self):
            self.hidden_size = 128
            self.intermediate_size = 256
            self.num_hidden_layers = 1
            self.num_attention_heads = 4
            self.num_key_value_heads = 4
            self.rms_norm_eps = 1e-6
            self.vocab_size = 1000
            self.max_position_embeddings = 2048
            self.head_dim = 32

    class DummyVllmConfig:
        def __init__(self):
            self.model_config = type('obj', (object,), {'hf_config': DummyConfig(), 'quantization': 'awq'})
            self.quant_config = AWQConfig()
            self.device = "cuda"

    with torch.device(device):
        model = LlamaModel(DummyVllmConfig())
    
    layer = model.layers[0]
    # Manual injection with stable values
    for p in [layer.self_attn.qkv_proj, layer.self_attn.o_proj, layer.mlp.gate_up_proj, layer.mlp.down_proj]:
        p.qweight = torch.randint(1, 10, (p.output_size, p.input_size // 8), device=device, dtype=torch.int32).contiguous()
        p.scales = torch.ones((p.output_size, p.input_size // 128), device=device, dtype=torch.float16).contiguous() * 0.01
        p.qzeros = torch.zeros((p.output_size, p.input_size // 128 // 8 + 1), device=device, dtype=torch.int32).contiguous()
        p.weight = torch.empty(1, device=device)
    
    # Ensure Norm weights are initialized
    layer.input_layernorm.weight.data.fill_(1.0)
    layer.post_attention_layernorm.weight.data.fill_(1.0)

    # Reference Forward (Standard Python path)
    x = torch.randn((1, 1, 128), device=device, dtype=torch.float16) * 0.1
    pos = torch.zeros((1, 1), device=device, dtype=torch.int64)
    with torch.inference_mode():
        ref_out = layer(x, pos, None, None)

    # Optimized Forward (Fast Dispatch path)
    layer.compile_fast_dispatch()
    with torch.inference_mode():
        opt_out = layer(x, pos, None, None)

    diff = (ref_out - opt_out).abs().max().item()
    print(f"Dispatch Logic Max Diff: {diff:.8f}")
    
    if diff < 1e-4:
        print("✅ PASS: Fast Dispatch logic is transparent.")
        return True
    else:
        print("❌ FAIL: Fast Dispatch results differ from standard path!")
        return False

if __name__ == "__main__":
    s1 = verify_awq_macro_fusion()
    s2 = verify_fast_dispatch_logic()
    
    if s1 and s2:
        print("\n🎉 ALL ACCURACY CHECKS PASSED.")
    else:
        exit(1)
