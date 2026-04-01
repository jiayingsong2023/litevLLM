# SPDX-License-Identifier: Apache-2.0
import os
import time
import torch
from vllm import LLM, SamplingParams

def run_64k_stress_test():
    model_path = "models/Qwen3.5-35B-AWQ"
    
    # 1. Generate a ~16k token prompt
    print(">>>> [1/4] Generating 16,000 token prompt...")
    base_phrase = "Artificial intelligence "
    prompt = base_phrase * 8000 # ~16k tokens
    
    # 2. Configure sampling
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=10,
    )

    # 3. Force Phase 3 optimized paths
    os.environ["FASTINFERENCE_KV_TYPE"] = "turbo_int4"
    os.environ["FASTINFERENCE_BLOCK_SIZE"] = "64"
    os.environ["FASTINFERENCE_KV_MAX_MODEL_LEN"] = "20000"
    os.environ["FASTINFERENCE_LITE_PREFILL_CHUNK"] = "4096"
    
    print(">>>> [2/4] Initializing LiteEngine (TurboQuant INT4 + BlockSize 64)...")
    try:
        llm = LLM(
            model=model_path,
            gpu_memory_utilization=0.70, # Conservative for extrapolation
            max_model_len=20000,
            trust_remote_code=True
        )

        # 4. Execute Inference
        print(">>>> [3/4] Starting 64k Context Prefill (Long Latency Expected)...")
        torch.cuda.reset_peak_memory_stats()
        start_inference = time.time()
        
        outputs = llm.generate([prompt], sampling_params)
        
        end_inference = time.time()
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        
        # 5. Report Results
        print("\n" + "="*50)
        print("64K CONTEXT STRESS TEST RESULTS")
        print("="*50)
        print(f"Status:           SUCCESS" if outputs else "Status:           FAILED")
        print(f"Total Latency:    {end_inference - start_inference:.2f}s")
        print(f"Peak VRAM Usage:  {peak_vram:.2f} GiB")
        
        if outputs:
            prompt_tokens = len(outputs[0].prompt_token_ids)
            generated_text = outputs[0].outputs[0].text
            print(f"Prompt Tokens:    {prompt_tokens}")
            print(f"Generated Text:   {generated_text[:100]}...")
        print("="*50)
        
    except Exception as e:
        print(f"\n!!!! Stress Test CRASHED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_64k_stress_test()
