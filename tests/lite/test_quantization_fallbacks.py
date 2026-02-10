# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.misc_utils import tensor_force_quant, quant_noise

def test_tensor_force_quant():
    assert tensor_force_quant(2) == True
    assert tensor_force_quant(1) == False

def test_quant_noise():
    x = torch.ones(10, 10)
    # With p=0, should be identical
    out = quant_noise(x, 0.0, 1)
    torch.testing.assert_close(x, out)
    
    # With p=1.0, should be different
    out = quant_noise(x, 1.0, 1)
    assert not torch.allclose(x, out)

@pytest.mark.parametrize("quant_type", [2, 12]) # Q4_0, Q4_K
def test_ggml_dequantize_fallback(quant_type):
    # Create a dummy quantized tensor
    # GGML Q4_0 uses 2-byte scale + 16 bytes for 32 nibbles = 18 bytes per block
    # For a row of 32 elements, it's 18 bytes.
    m, n = 1, 32
    if quant_type == 2:
        w = torch.zeros((m, 18), dtype=torch.uint8)
    else:
        # Q4_K uses a different size
        import gguf
        block_size, type_size = gguf.GGML_QUANT_SIZES[quant_type]
        w = torch.zeros((m, (n // block_size) * type_size), dtype=torch.uint8)
        
    out = ops.ggml_dequantize(w, quant_type, m, n, torch.float16)
    assert out.shape == (m, n)
    assert out.dtype == torch.float16

def test_scaled_fp8_quant_fallback():
    x = torch.randn(16, 128, dtype=torch.float16, device="cuda")
    out, scale = ops.scaled_fp8_quant(x)
    
    assert out.shape == x.shape
    assert out.dtype in [torch.float8_e4m3fn, torch.float8_e4m3fnuz]
    assert scale.numel() == 1
    
    # Test per-token
    out, scale = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
    assert scale.shape == (16, 1)

if __name__ == "__main__":
    # Simple manual run if pytest is not available
    test_tensor_force_quant()
    print("test_tensor_force_quant passed")
    test_quant_noise()
    print("test_quant_noise passed")
    if torch.cuda.is_available():
        test_scaled_fp8_quant_fallback()
        print("test_scaled_fp8_quant_fallback passed")
