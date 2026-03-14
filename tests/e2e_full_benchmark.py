# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
import torch
import os
from vllm import LLM, SamplingParams
from vllm.config import VllmConfig, ModelConfig, LoadConfig
from vllm.engine.async_llm import AsyncLLM

MODELS = [
    ("models/TinyLlama-1.1B-Chat-v1.0", "TinyLlama-1.1B (Dense)"),
    ("models/Qwen3.5-9B-GGUF", "Qwen3.5-9B (GGUF Q4)"),
    ("models/Qwen3.5-9B-AWQ", "Qwen3.5-9B (AWQ INT4)"),
    ("models/DeepSeek-V2-Lite-GGUF", "DeepSeek-V2-Lite (MoE)"),
    ("models/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash (MoE)"),
    ("models/Qwen3.5-35B-MoE-GGUF", "Qwen3.5-35B (Large MoE)"),
]

async def run_benchmark(model_path, model_name):
    # Determine load parameters based on model
    is_35b = "35B" in model_name
    batch_size = 1 if is_35b else 8
    max_new_tokens = 128 # We test generation throughput
    # We use a long prompt to simulate "content length"
    content_len = 1024 if is_35b else 2048
    
    print(f"\n{'='*60}")
    print(f"BENCHMARKING: {model_name}")
    print(f"LOAD: BS={batch_size}, Target Content Length={content_len}")
    print(f"PATH: {model_path}")
    if torch.cuda.is_available():
        print(f"INITIAL MEMORY: {torch.cuda.memory_allocated()/1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    print(f"{'='*60}")
    
    if not os.path.exists(model_path):
        print(f"Skipping {model_name}: Path not found.")
        return

    try:
        # 1. Config Setup (optimized for AMD AI Max)
        m_cfg = ModelConfig(
            model=model_path,
            tokenizer=model_path,
            tokenizer_mode="auto",
            trust_remote_code=True,
            dtype="float16",
            max_model_len=max(4096, content_len + max_new_tokens)
        )
        
        from vllm.config import CacheConfig, SchedulerConfig, LoadConfig
        v_cfg = VllmConfig(
            model_config=m_cfg,
            cache_config=CacheConfig(block_size=16, gpu_memory_utilization=0.92, swap_space=0),
            scheduler_config=SchedulerConfig(max_num_batched_tokens=content_len * batch_size, 
                                           max_num_seqs=batch_size, 
                                           max_model_len=m_cfg.max_model_len),
            load_config=LoadConfig(load_format="auto"),
            quant_config=None,
        )
        if any(f.endswith(".gguf") for f in os.listdir(model_path)):
            from vllm.model_executor.layers.quantization.gguf import GGUFConfig
            v_cfg.quant_config = GGUFConfig()

        # 2. Load Entrypoint
        llm = AsyncLLM.from_vllm_config(v_cfg)
        sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
        
        # Construct long prompt to fill content length
        base_prompt = "Tell me a long story about AI and the future of human civilization. "
        repeat_factor = (content_len // 10) # rough estimate
        long_prompt = base_prompt * (repeat_factor // len(base_prompt.split()) + 1)
        
        # 3. Benchmark - Warmup
        print(f"[{model_name}] Warming up with single request...")
        async for _ in llm.generate("Hello", sampling_params, "warmup"): pass
        
        # 4. Benchmark - Full Concurrent Run
        print(f"[{model_name}] Running concurrent test (BS={batch_size})...")
        
        async def run_single_request(idx):
            req_id = f"bench_{idx}_{int(time.time())}"
            tokens = 0
            async for output in llm.generate(long_prompt, sampling_params, req_id):
                if output.finished:
                    tokens = len(output.outputs[0].token_ids)
            return tokens

        start_time = time.perf_counter()
        # Launch concurrent requests
        results = await asyncio.gather(*[run_single_request(i) for i in range(batch_size)])
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        total_tokens = sum(results)
        tps = total_tokens / duration
        
        print(f"  🏆 RESULT: {tps:.2f} aggregate tokens/sec")
        print(f"  📊 Stats: Total Tokens: {total_tokens}, Total Time: {duration:.2f}s, Avg Latency: {duration*1000/(total_tokens/batch_size):.2f} ms/tok/user")
        
        # Cleanup
        llm.shutdown()
        if hasattr(llm, "engine"):
            if hasattr(llm.engine, "model"): del llm.engine.model
            del llm.engine
        del llm
        from vllm.model_executor.layers.quantization.tensor import clear_global_weight_cache
        clear_global_weight_cache()
        import gc; gc.collect()
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback; traceback.print_exc()

async def main():
    print("================================================================")
    print("FASTINFERENCE REAL-WEIGHT END-TO-END BENCHMARK (AMD AI MAX+395)")
    print("================================================================")
    
    for path, name in MODELS:
        await run_benchmark(path, name)

if __name__ == "__main__":
    asyncio.run(main())
