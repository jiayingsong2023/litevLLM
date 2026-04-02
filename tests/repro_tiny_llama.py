import torch
import os
from vllm.model_executor.model_loader import get_model
from vllm.config import VllmConfig, ModelConfig, CacheConfig, SchedulerConfig, LoadConfig
from vllm.engine.lite_engine import LiteEngine
from vllm.sampling_params import SamplingParams

def test_tiny_llama():
    model_path = "models/TinyLlama-1.1B-Chat-v1.0"
    if not os.path.exists(model_path):
        print(f"Model path {model_path} not found.")
        return

    # Force INT4 from env if provided, else default
    if "FASTINFERENCE_KV_TYPE" not in os.environ:
         os.environ["FASTINFERENCE_KV_TYPE"] = "turbo_int4"
    
    # Configure and Load
    mc = ModelConfig(model=model_path, tokenizer=model_path, tokenizer_mode="auto", trust_remote_code=True, max_model_len=1024)
    cc = CacheConfig(block_size=16, gpu_memory_utilization=0.9, swap_space=4, cache_dtype="auto")
    sc = SchedulerConfig(max_num_batched_tokens=2048, max_num_seqs=4, max_model_len=1024)
    lc = LoadConfig(load_format="auto")
    
    vc = VllmConfig(model_config=mc, cache_config=cc, scheduler_config=sc, load_config=lc)
    engine = LiteEngine(vc)
    
    if engine.tokenizer is None:
        from transformers import AutoTokenizer
        engine.tokenizer = AutoTokenizer.from_pretrained(model_path)

    prompt = "Hello, how are you?"
    params = SamplingParams(temperature=0.0, max_tokens=20)
    
    print(f"\n--- Testing Prompt: {prompt} ---")
    engine.add_request("req1", prompt, params)
    
    for _ in range(30):
        outputs = engine.step()
        if not outputs:
            break
        for out in outputs:
            # RequestOutput has outputs list containing CompletionOutput
            if out.outputs:
                text = out.outputs[0].text
                # Display only the new tokens if needed, but for repro just print text
                print(f" (Step: {text})", flush=True)
            if out.finished:
                print(f"\nFinished: {out.request_id}")
                return

if __name__ == "__main__":
    test_tiny_llama()
