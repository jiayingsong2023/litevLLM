# SPDX-License-Identifier: Apache-2.0
import torch
import time
from vllm.model_executor.layers.quantization.awq import AWQConfig

def test_awq_triton_perf():
    device = "cuda"
    dtype = torch.float16
    
    # Standard Llama-7B Layer Dims
    hidden_size = 4096
    intermediate_size = 11008
    group_size = 128
    
    config = AWQConfig(weight_bits=4, group_size=group_size, zero_point=True)
    
    # Create a mock layer
    class MockLayer:
        def __init__(self):
            self.output_size = intermediate_size
            self.bias = None
            config.init_layer(self)
            # Fill with mock quantized data
            self.qweight = torch.randint(0, 100, (hidden_size, intermediate_size // 8), device=device, dtype=torch.int32)
            self.qzeros = torch.randint(0, 100, (hidden_size // group_size, intermediate_size // 8), device=device, dtype=torch.int32)
            self.scales = torch.randn((hidden_size // group_size, intermediate_size), device=device, dtype=dtype)

    layer = MockLayer()
    
    print(f"=== AWQ Triton Performance Test ({hidden_size}x{intermediate_size}) ===")
    
    for bs in [1, 32]:
        x = torch.randn((bs, hidden_size), device=device, dtype=dtype)
        
        # Warmup
        for _ in range(10):
            _ = config.apply(layer, x)
        torch.cuda.synchronize()
        
        # Benchmark
        iters = 100
        start = time.time()
        for _ in range(iters):
            _ = config.apply(layer, x)
        torch.cuda.synchronize()
        end = time.time()
        
        latency = (end - start) / iters * 1000
        tps = (1000 / latency) * bs
        print(f"Batch Size {bs:2}: Latency = {latency:6.2f} ms, TPS = {tps:8.2f}")

if __name__ == "__main__":
    test_awq_triton_perf()
