import os
import subprocess
import time
import gc
import torch
from vllm import LLM, SamplingParams

def cleanup_gpu():
    print("\n" + "="*50)
    print(">>> [INIT] Cleaning GPU Environment...")
    print("="*50)
    # Clear torch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Force kill any lingering python processes using GPU (except current one)
    # Note: Using shell to find and kill is risky, but we'll try basic cleanup
    # For now, just rely on GC and empty_cache as we're running in one process
    # but sequentially initializing and deleting LLM objects.
    time.sleep(2)
    
    # Check VRAM via rocm-smi
    try:
        res = subprocess.run(["rocm-smi", "--showmeminfo", "vram"], capture_output=True, text=True)
        if res.returncode == 0:
            print(res.stdout)
    except Exception as e:
        print(f"Cleanup warning: {e}")

def run_benchmark(name, model_path, batch_size=32, context_len=4096, is_moe=False):
    print(f"\n>>> [RUNNING] {name} (BS={batch_size}, Context={context_len})...")
    
    try:
        # Configuration for AMD AI MAX+395 (60GB VRAM)
        # Using dynamic expert cache for MoE models
        if is_moe:
            os.environ["FASTINFERENCE_MOE_LRU_SIZE"] = "32" # Match user request for balanced mode
        
        llm = LLM(
            model=model_path,
            max_num_seqs=batch_size,
            max_model_len=context_len,
            trust_remote_code=True,
            gpu_memory_utilization=0.9, # Leave some room for overhead
            enforce_eager=True,
            kv_cache_dtype="fp8" # User requested FP8 path
        )
        
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=128, # Generate some tokens for throughput measurement
            ignore_eos=True
        )
        
        # Prepare dummy prompts
        prompts = ["Hello, what can you do? " * (context_len // 10)] * batch_size
        
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        end_time = time.time()
        
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        elapsed = end_time - start_time
        tps = total_tokens / elapsed
        
        print(f">>> [RESULT] {name}: {tps:.2f} tokens/sec")
        
        # Cleanup LLM to free VRAM for next model
        del llm
        cleanup_gpu()
        
        return True, f"{tps:.2f} t/s"
    except Exception as e:
        print(f">>> [FAILED] {name}: {str(e)}")
        cleanup_gpu()
        return False, str(e)

def main():
    print(f"Starting Full Regression Suite: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # 1. TinyLlama Baseline
    success, metric = run_benchmark("TinyLlama-1.1B", "models/TinyLlama-1.1B-Chat-v1.0-GGUF", batch_size=1, context_len=1024)
    results.append({"Task": "TinyLlama Baseline", "Status": "PASS" if success else "FAIL", "Metric": metric})
    
    # 2. DeepSeek-V2-Lite (MoE)
    success, metric = run_benchmark("DeepSeek-V2-Lite", "models/DeepSeek-V2-Lite-GGUF", batch_size=32, context_len=4096, is_moe=True)
    results.append({"Task": "DeepSeek-V2-Lite", "Status": "PASS" if success else "FAIL", "Metric": metric})
    
    # 3. GLM-4.7-Flash (MoE)
    # Note: User mentioned BS=16 might be better for 60GB VRAM at 4096 context
    success, metric = run_benchmark("GLM-4.7-Flash", "models/GLM-4.7-9B-Flash-GGUF", batch_size=16, context_len=4096, is_moe=True)
    results.append({"Task": "GLM-4.7-Flash", "Status": "PASS" if success else "FAIL", "Metric": metric})
    
    # 4. Qwen-3.5-9B
    success, metric = run_benchmark("Qwen-3.5-9B", "models/Qwen-3.5-9B-GGUF", batch_size=32, context_len=4096)
    results.append({"Task": "Qwen-3.5-9B", "Status": "PASS" if success else "FAIL", "Metric": metric})
    
    # 5. Qwen-3.5-35B-MoE
    # User suggested BS=8, Context=1024 for 35B
    success, metric = run_benchmark("Qwen-3.5-35B-MoE", "models/Qwen-3.5-35B-MoE-GGUF", batch_size=8, context_len=1024, is_moe=True)
    results.append({"Task": "Qwen-3.5-35B-MoE", "Status": "PASS" if success else "FAIL", "Metric": metric})

    # --- FINAL REPORT ---
    print("\n" + "="*80)
    print(f"FINAL REGRESSION REPORT - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print(f"| {'Task':<30} | {'Status':<10} | {'Metric':<20} |")
    print(f"|{'-'*32}|{'-'*12}|{'-'*22}|")
    for r in results:
        print(f"| {r['Task']:<30} | {r['Status']:<10} | {r['Metric']:<20} |")
    print("="*80)

    if all(r["Status"] == "PASS" for r in results):
        print("\nAll regression tests passed successfully.")
        return 0
    else:
        print("\nSome regression tests failed. Check output logs.")
        return 1

if __name__ == "__main__":
    exit(main())
