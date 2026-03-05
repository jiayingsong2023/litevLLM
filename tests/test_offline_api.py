# SPDX-License-Identifier: Apache-2.0
from vllm import LLM, SamplingParams
import torch
import os

def test_model(model_path, name):
    print(f"\n>>> TESTING MODEL: {name} ({model_path})")
    try:
        # 1. Initialize
        llm = LLM(model=model_path)
        sampling_params = SamplingParams(max_tokens=16, temperature=0.7)
        
        # 2. Generate
        prompts = ["Tell me a short story about an AI."]
        print(f"[{name}] Requesting Generation...")
        outputs = llm.generate(prompts, sampling_params)
        
        # 3. Print
        for output in outputs:
            print(f"[{name}] Result: {output.outputs[0].text!r}")
            print("-" * 20)
            
        # 4. Clean up
        del llm
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"!!! {name} Test Failed: {e}")
        import traceback; traceback.print_exc()
        return False

if __name__ == "__main__":
    models = [
        ("models/Qwen3.5-9B-GGUF", "Qwen3.5-9B-GGUF"),
        ("models/DeepSeek-V2-Lite-GGUF", "DeepSeek-V2-GGUF"),
        ("models/Qwen3.5-9B-AWQ", "Qwen3.5-9B-AWQ") # THE FINAL CHALLENGE
    ]
    
    for path, name in models:
        if os.path.exists(path):
            test_model(path, name)
        else:
            print(f"Skipping {name}, path not found: {path}")
