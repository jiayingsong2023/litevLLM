
import torch
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.config import VllmConfig
from vllm.config.vllm import set_current_vllm_config

def test_rms_norm():
    print("Testing RMSNorm with Triton Kernel...")
    if not torch.cuda.is_available():
        print("Skipping test: CUDA not available")
        return

    # Setup dummy config
    config = VllmConfig()
    
    with set_current_vllm_config(config):
        torch.manual_seed(0)
        
        batch_size = 4
        hidden_size = 1024
        epsilon = 1e-6
        dtype = torch.float16
        device = "cuda"

        # Create Layer
        layer = RMSNorm(hidden_size, eps=epsilon).to(device=device, dtype=dtype)
        
        # Input
        x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        
        # 1. Native PyTorch Forward
        # Note: RMSNorm.forward_native calls forward_static
        out_native = layer.forward_native(x)
        
        # 2. Triton Forward (via forward_cuda -> _custom_ops.rms_norm -> triton_rms_norm)
        out_triton = layer.forward_cuda(x)

        # Compare
        # RMSNorm forward_native returns (output, residual) tuple if residual is passed, or just output
        # checking signature: forward_native(x, residual=None) -> Tensor | tuple
        
        if isinstance(out_native, tuple):
            out_native = out_native[0]
        if isinstance(out_triton, tuple):
            out_triton = out_triton[0]

        diff = (out_native - out_triton).abs().max()
        print(f"RMSNorm Max Diff: {diff}")
        assert diff < 1e-3, f"Mismatch! Max diff: {diff}"
        print("RMSNorm Passed!")

def test_fused_add_rms_norm():
    print("\nTesting Fused Add RMSNorm with Triton Kernel...")
    if not torch.cuda.is_available():
        print("Skipping test: CUDA not available")
        return

    # Setup dummy config
    config = VllmConfig()

    with set_current_vllm_config(config):
        torch.manual_seed(0)
        
        batch_size = 4
        hidden_size = 1024
        epsilon = 1e-6
        dtype = torch.float16
        device = "cuda"

        # Create Layer
        layer = RMSNorm(hidden_size, eps=epsilon).to(device=device, dtype=dtype)
        
        # Input
        x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        residual = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        
        # Clone for native baseline (since inplace updates happen)
        x_native = x.clone()
        res_native = residual.clone()
        
        x_triton = x.clone()
        res_triton = residual.clone()

        # 1. Native PyTorch Forward
        out_native, new_res_native = layer.forward_native(x_native, res_native)
        
        # 2. Triton Forward
        out_triton, new_res_triton = layer.forward_cuda(x_triton, res_triton)

        # Check Output
        diff_out = (out_native - out_triton).abs().max()
        print(f"Fused Output Max Diff: {diff_out}")
        
        # Check Residual (Should be x + residual)
        diff_res = (new_res_native - new_res_triton).abs().max()
        print(f"Fused Residual Max Diff: {diff_res}")

        assert diff_out < 1e-3, f"Output Mismatch! {diff_out}"
        assert diff_res < 1e-3, f"Residual Mismatch! {diff_res}"
        
        print("Fused Add RMSNorm Passed!")

if __name__ == "__main__":
    try:
        test_rms_norm()
        test_fused_add_rms_norm()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Test Failed with error: {e}")
