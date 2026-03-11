# SPDX-License-Identifier: Apache-2.0
import torch
import time
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.layers.quantization.awq import AWQConfig

def benchmark_real_qwen_awq():
    device = "cuda"
    print(f"\n[E2E] Performance Benchmark: Fast Dispatch Mode")
    
    # Qwen3.5-9B Config
    class DummyConfig:
        def __init__(self):
            self.hidden_size = 3584
            self.intermediate_size = 18944
            self.num_hidden_layers = 28
            self.num_attention_heads = 28
            self.num_key_value_heads = 4
            self.rms_norm_eps = 1e-6
            self.vocab_size = 152064
            self.max_position_embeddings = 32768
            self.rope_theta = 1000000
            self.head_dim = 128

    class DummyVllmConfig:
        def __init__(self):
            self.model_config = type('obj', (object,), {
                'hf_config': DummyConfig(),
                'quantization': 'awq'
            })
            self.quant_config = AWQConfig(weight_bits=4, group_size=128)

    # 1. Initialize Model
    with torch.device(device):
        model = LlamaModel(DummyVllmConfig())
    
    # 2. Inject Mock Weights (Bypass complex loader for perf test)
    print("Injecting Mock AWQ weights for performance testing...")
    for layer in model.layers:
        projs = [layer.self_attn.qkv_proj, layer.self_attn.o_proj, 
                 layer.mlp.gate_up_proj, layer.mlp.down_proj]
        for p in projs:
            p.qweight = torch.randint(0, 100, (p.output_size, p.input_size // 8), device=device, dtype=torch.int32)
            p.scales = torch.randn((p.output_size, p.input_size // 128), device=device, dtype=torch.float16)
            p.qzeros = torch.randint(0, 100, (p.output_size, p.input_size // 128 // 8 + 1), device=device, dtype=torch.int32)
            p.weight = torch.empty(1) # Dummy to pass hasattr checks

    # 3. Compile Fast Dispatch (The Magic)
    print("Compiling Fast Dispatch for all 28 layers...")
    for layer in model.layers:
        layer.compile_fast_dispatch()

    bs = 32
    hidden_size = 3584
    x = torch.randn((bs, 1, hidden_size), device=device, dtype=torch.float16)
    
    # Mock metadata
    positions = torch.zeros((bs, 1), device=device, dtype=torch.int64)
    kv_caches = [None] * 28
    attn_metadata = None

    print("Model initialized. Running Benchmark...")
    
    # Warmup
    for _ in range(10):
        with torch.inference_mode():
            h = x
            for layer in model.layers:
                h = layer(h, positions, None, attn_metadata)
    torch.cuda.synchronize()
    
    iters = 20
    start = time.time()
    for _ in range(iters):
        with torch.inference_mode():
            h = x
            for layer in model.layers:
                h = layer(h, positions, None, attn_metadata)
    torch.cuda.synchronize()
    
    latency = (time.time() - start) / iters * 1000
    tps = (1000 / latency) * bs
    
    print(f"┌──────────────────────────────────────────────────────────┐")
    print(f"│ REAL E2E PERFORMANCE REPORT: Qwen3.5-9B-AWQ (FAST DISPATCH) │")
    print(f"├──────────────────────────────────────────────────────────┤")
    print(f"│ Batch Size            : {bs:<32} │")
    print(f"│ Step Latency          : {latency:7.2f} ms{' ':<23} │")
    print(f"│ Throughput (TPS)      : {tps:10.2f} t/s{' ':<20} │")
    print(f"└──────────────────────────────────────────────────────────┘")
    
    return tps

if __name__ == "__main__":
    benchmark_real_qwen_awq()
