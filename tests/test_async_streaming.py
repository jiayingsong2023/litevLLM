# SPDX-License-Identifier: Apache-2.0
import asyncio
import torch
import os
from vllm.engine.async_llm import AsyncLLM
from vllm.sampling_params import SamplingParams
from vllm.config import VllmConfig

async def test_async_streaming():
    print("=== STARTING ASYNC STREAMING TEST ===")
    model_path = "models/TinyLlama-1.1B-Chat-v1.0"
    
    # Use real AutoConfig to get dimensions
    from transformers import AutoConfig
    hf_cfg = AutoConfig.from_pretrained(model_path)
    
    class LiteVllmConfig:
        def __init__(self, path, hf_cfg):
            n_heads = getattr(hf_cfg, "num_attention_heads", 32)
            n_kv_heads = getattr(hf_cfg, "num_key_value_heads", n_heads)
            h_size = getattr(hf_cfg, "hidden_size", 4096)
            n_layers = getattr(hf_cfg, "num_hidden_layers", 32)
            
            self.model_config = type('obj', (object,), {
                'hf_config': hf_cfg, 'dtype': torch.float16,
                'max_model_len': 2048, 'model': path,
                'get_num_kv_heads': lambda x: n_kv_heads,
                'get_head_size': lambda: h_size // n_heads,
                'get_num_layers': lambda x: n_layers,
            })
            self.quant_config = None
            self.parallel_config = type('obj', (object,), {'tensor_parallel_size': 1, 'world_size': 1})
    
    v_cfg = LiteVllmConfig(model_path, hf_cfg)
    print(">>> Initializing AsyncLLM...")
    engine = AsyncLLM(v_cfg)
    
    sampling_params = SamplingParams(max_tokens=10)
    print(">>> Requesting Streaming Generation...")
    
    full_text = ""
    async for output in engine.generate("The meaning of life is", sampling_params, "test-request"):
        if not output.outputs: continue
        new_text = output.outputs[0].text
        delta = new_text[len(full_text):]
        full_text = new_text
        if delta: print(f"Chunk: {delta!r}")
            
    print(f"\n>>> Final Full Text: {full_text!r}")
    print("=== ASYNC STREAMING TEST PASSED ===")
    engine.shutdown()

if __name__ == "__main__":
    asyncio.run(test_async_streaming())
