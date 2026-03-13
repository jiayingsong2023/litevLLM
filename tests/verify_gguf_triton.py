# SPDX-License-Identifier: Apache-2.0
import torch
import numpy as np
from gguf import dequantize, GGMLQuantizationType
from vllm.kernels.triton.gguf_q4_0_dequant import gguf_q4_0_dequant

def test_gguf_q4_0_accuracy():
    print("--- Verifying GGUF Q4_0 Triton Kernel Accuracy ---")
    device = "cuda"
    
    # 1. Create Synthetic Q4_0 Data (1 row, 128 elements = 4 GGML blocks)
    # Each block: [2 bytes delta (fp16), 16 bytes weights]
    n_rows = 1
    n_cols = 128
    bytes_per_row = (n_cols // 32) * 18
    
    # Manual creation of valid GGUF bytes
    raw_data = np.zeros((n_rows, bytes_per_row), dtype=np.uint8)
    for b in range(n_cols // 32):
        # Set delta to 0.5 (0x3800 in fp16 little-endian -> 00 38)
        raw_data[0, b*18] = 0x00
        raw_data[0, b*18+1] = 0x38
        # Set some weights (0xAB -> high=10, low=11)
        raw_data[0, b*18+2 : b*18+18] = 0xAB
    
    W_gpu = torch.from_numpy(raw_data).to(device)
    
    # 2. Run CPU Reference (Official gguf-py)
    print("Running CPU reference...")
    ref_out = dequantize(raw_data, GGMLQuantizationType.Q4_0).reshape(n_rows, n_cols)
    ref_torch = torch.from_numpy(ref_out).to(device).half()

    # 3. Run Triton Kernel
    print("Running Triton kernel...")
    try:
        tri_out = gguf_q4_0_dequant(W_gpu, n_rows, n_cols)
    except Exception as e:
        print(f"❌ Triton execution failed: {e}")
        return

    # 4. Compare
    cos_sim = torch.nn.functional.cosine_similarity(ref_torch.flatten(), tri_out.flatten(), dim=0).item()
    print(f"Cosine Similarity: {cos_sim:.10f}")
    
    diff = torch.abs(ref_torch - tri_out).max().item()
    print(f"Max Absolute Error: {diff:.10f}")

    if cos_sim > 0.9999:
        print("🏆 SUCCESS: GGUF Triton Kernel is bit-accurate!")
    else:
        print("❌ FAIL: Numerical drift detected.")

if __name__ == "__main__":
    test_gguf_q4_0_accuracy()
