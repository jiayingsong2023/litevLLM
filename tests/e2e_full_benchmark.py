# SPDX-License-Identifier: Apache-2.0
import torch
import time
import os
import gc
from vllm.model_executor.model_loader import get_model
from vllm.config import VllmConfig, ModelConfig, LoadConfig, CacheConfig, SchedulerConfig

def benchmark_real_model(name, model_path, batch_size=32, context_len=4096, env_overrides=None):
    print(f"\n>>> [REAL] Benchmarking: {name}")
    print(f">>> Path: {model_path}, BS: {batch_size}, Ctx: {context_len}")
    
    if env_overrides:
        os.environ.update(env_overrides)

    try:
        # 1. Config (STRICT SIGNATURE MATCHING)
        m_config = ModelConfig(model=model_path, tokenizer=model_path, trust_remote_code=True, dtype="float16", max_model_len=context_len)
        c_config = CacheConfig(block_size=16, gpu_memory_utilization=0.9, swap_space=4, cache_dtype="auto")
        s_config = SchedulerConfig(max_num_batched_tokens=max(batch_size * 128, 4096), max_num_seqs=batch_size, max_model_len=context_len)
        l_config = LoadConfig(load_format="gguf")
        
        v_config = VllmConfig(model_config=m_config, cache_config=c_config, scheduler_config=s_config, load_config=l_config)
        
        # 2. Load
        print("Loading weights...")
        model = get_model(v_config)
        
        # 3. Inputs
        input_ids = torch.randint(0, 32000, (batch_size, 128), device="cuda")
        positions = torch.arange(128, device="cuda").unsqueeze(0).repeat(batch_size, 1)
        
        # Pre-allocate real KV caches for prefill
        num_layers = m_config.hf_config.num_hidden_layers
        num_kv_heads = getattr(m_config.hf_config, "num_key_value_heads", m_config.hf_config.num_attention_heads)
        head_size = m_config.hf_config.hidden_size // m_config.hf_config.num_attention_heads
        kv_dtype = torch.float8_e4m3fn if os.environ.get("FASTINFERENCE_KV_FP8", "1") == "1" else torch.float16
        
        kv_caches = []
        for _ in range(num_layers):
            k = torch.zeros((batch_size, context_len, num_kv_heads, head_size), device="cuda", dtype=kv_dtype)
            v = torch.zeros_like(k)
            kv_caches.append((k, v))
            
        attn_metadata = {
            "slot_mapping": torch.arange(batch_size, device="cuda", dtype=torch.long),
            "seq_lens": torch.ones(batch_size, device="cuda", dtype=torch.int32) * 128,
            "kv_start_indices": torch.zeros(batch_size, device="cuda", dtype=torch.int32),
            "is_prefill": True
        }

        # 4. Warmup
        print("Warmup...")
        for _ in range(3):
            with torch.inference_mode():
                model(input_ids, positions, kv_caches, attn_metadata)
        
        # 5. Profile (Full context)
        full_input = torch.randint(0, 32000, (batch_size, context_len), device="cuda")
        full_positions = torch.arange(context_len, device="cuda").unsqueeze(0).repeat(batch_size, 1)
        attn_metadata["seq_lens"] = torch.ones(batch_size, device="cuda", dtype=torch.int32) * context_len
        
        print("Benchmarking...")
        torch.cuda.synchronize()
        start = time.perf_counter()
        iters = 2
        for i in range(iters):
            print(f"  Iteration {i+1}/{iters}...")
            with torch.inference_mode():
                model(full_input, full_positions, kv_caches, attn_metadata)
            torch.cuda.synchronize()
        end = time.perf_counter()
        
        avg_latency = (end - start) / iters
        tps = (batch_size * context_len) / avg_latency
        print(f"RESULT: Latency={avg_latency*1000:.2f}ms, Throughput={tps:.2f} tokens/sec")
        
    except Exception as e:
        print(f"Failed to benchmark {name}: {e}")
        import traceback; traceback.print_exc()
    finally:
        if 'model' in locals(): del model
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    benchmark_real_model("TinyLlama-1.1B", "models/TinyLlama-1.1B-Chat-v1.0", batch_size=32, context_len=4096)
