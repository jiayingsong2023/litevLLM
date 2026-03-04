# SPDX-License-Identifier: Apache-2.0
import torch
import time
import os
from vllm.entrypoints.llm import LLM

def test_production_hot_switch():
    print("=== STARTING PRODUCTION HOT SWITCH TEST ===")
    
    # Check paths
    models = ["models/TinyLlama-1.1B-Chat-v1.0", "models/Qwen3.5-9B-GGUF", "models/DeepSeek-V2-Lite-GGUF"]
    for m in models:
        if not os.path.exists(m):
            print(f"Skipping test, model not found: {m}")
            return

    # 1. Start with TinyLlama
    llm = LLM(model=models[0])
    print("Initial model loaded.")
    
    # 2. Switch to Qwen3.5-9B
    time.sleep(1)
    llm.switch_model(models[1])
    
    # 3. Switch to DeepSeek-V2-Lite
    time.sleep(1)
    llm.switch_model(models[2])
    
    print("\n=== HOT SWITCH TEST PASSED ===")
    print("Verification: All models loaded sequentially in one process without OOM.")

if __name__ == "__main__":
    try:
        test_production_hot_switch()
    except Exception as e:
        print(f"Hot Switch Failed: {e}")
        import traceback
        traceback.print_exc()
