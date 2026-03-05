# SPDX-License-Identifier: Apache-2.0
import torch
import time
import os
import gc
from vllm import LLM, SamplingParams
from vllm.model_executor.layers.quantization.gguf import clear_gguf_cache

def run_benchmark(model_path, name, batch_size=32):
    print(f"\n>>> [BENCHMARK] Testing {name}...")
    try:
        # 1. Purge VRAM
        clear_gguf_cache()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        # 2. Load Model
        start_load = time.time()
        llm = LLM(model=model_path)
        load_time = time.time() - start_load
        
        # 3. Setup Params - IMPORTANT: Using single token prompts for BS=32 throughput
        sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
        prompts = ["."] * batch_size # Single token prompts
        
        # 4. Warmup
        for _ in range(2):
            llm.generate(prompts, sampling_params)
        torch.cuda.synchronize()
        
        # 5. Iterative Test
        iters = 10
        start_time = time.time()
        for _ in range(iters):
            llm.generate(prompts, sampling_params)
        torch.cuda.synchronize()
        end_time = time.time()
        
        # 6. Metrics
        avg_latency = (end_time - start_time) / iters * 1000
        tps = (1000 / avg_latency) * batch_size
        vram_max = torch.cuda.max_memory_allocated() / 1024**3
        
        print(f"[{name}] Result: {tps:.2f} tokens/sec | Latency: {avg_latency:.2f}ms | VRAM: {vram_max:.2f}GB")
        
        del llm
        return tps
    except Exception as e:
        print(f"!!! {name} Benchmark Failed: {e}")
        import traceback; traceback.print_exc()
        return 0.0

if __name__ == "__main__":
    test_suite = [
        ("models/TinyLlama-1.1B-Chat-v1.0", "TinyLlama-1.1B (Dense FP16)"),
        ("models/Qwen3.5-9B-GGUF", "Qwen3.5-9B (GGUF)"),
        ("models/Qwen3.5-9B-AWQ", "Qwen3.5-9B (AWQ Real)"),
        ("models/DeepSeek-V2-Lite-GGUF", "DeepSeek-V2-Lite (MoE GGUF)")
    ]
    
    results = {}
    print("="*60)
    print("FASTINFERENCE V1.0 PERFORMANCE FINAL AUDIT")
    print("Hardware: AMD AI Max 395 (60GB VRAM / 128GB RAM)")
    print("Batch Size: 32")
    print("="*60)
    
    for path, name in test_suite:
        if os.path.exists(path):
            tps = run_benchmark(path, name, batch_size=32)
            results[name] = tps
        else:
            print(f"Skipping {name} (Path not found)")

    print("\n" + "="*45)
    print(f"{'MODEL NAME':<30} | {'TPS':<10}")
    print("-" * 45)
    for name, tps in results.items():
        print(f"{name:<30} | {tps:>8.2f} t/s")
    print("="*45)
