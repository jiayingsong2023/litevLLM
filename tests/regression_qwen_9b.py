import os
import torch
import time
from vllm.model_executor.model_loader import get_model
from vllm.transformers_utils.config import get_config
from vllm.model_executor.utils import set_random_seed

def benchmark_qwen_9b_fp8(model_path, batch_size, context_len=4096):
    print(f"\n>>> [Regression] Qwen-3.5-9B: BS={batch_size}, Context={context_len}, FP8 (Balanced Mode)")
    
    # FP8 Configuration
    os.environ["FASTINFERENCE_DEEPSEEK_FP8"] = "1"
    os.environ["FASTINFERENCE_MOE_CACHE_MODE"] = "dynamic"
    os.environ["FASTINFERENCE_MOE_LRU_SIZE"] = "32"
    
    v_config = get_config(model_path, trust_remote_code=True)
    v_config.architectures = ["Qwen2ForCausalLM"]
    
    print("Loading model weights (GGUF + Optimized Paths)...")
    model = get_model(v_config).cuda().half()
    
    # Input preparation
    input_ids = torch.randint(0, 1000, (batch_size, context_len), device="cuda")
    positions = torch.arange(context_len, device="cuda").unsqueeze(0).expand(batch_size, -1)
    
    # Dummy metadata
    from vllm.model_executor.layers.attention import AttentionMetadata
    num_blocks = (batch_size * context_len) // 16 + 1024
    kv_caches = []
    for _ in range(v_config.num_hidden_layers):
        kv_caches.append((torch.zeros((num_blocks, 16, v_config.num_key_value_heads, 128), device="cuda", dtype=torch.float16),
                         torch.zeros((num_blocks, 16, v_config.num_key_value_heads, 128), device="cuda", dtype=torch.float16)))
    
    attn_metadata = AttentionMetadata(
        num_prefills=batch_size, num_prefill_tokens=batch_size * context_len,
        num_decode_tokens=0, slot_mapping=torch.arange(batch_size * context_len, device="cuda", dtype=torch.int32),
        seq_lens=torch.full((batch_size,), context_len, device="cuda", dtype=torch.int32),
        max_prefill_seq_len=context_len, max_decode_seq_len=0,
        block_tables=torch.zeros((batch_size, num_blocks // batch_size), device="cuda", dtype=torch.int32),
        use_cuda_graph=False
    )

    print(f"Warmup (Balanced Mode)...")
    with torch.inference_mode():
        model(input_ids, positions, kv_caches, attn_metadata)
    torch.cuda.synchronize()

    print("Benchmarking...")
    start_time = time.time()
    iters = 3
    for _ in range(iters):
        with torch.inference_mode():
            model(input_ids, positions, kv_caches, attn_metadata)
    torch.cuda.synchronize()
    end_time = time.time()

    avg_latency = (end_time - start_time) / iters
    total_tokens = batch_size * context_len
    tps = total_tokens / avg_latency
    
    print(f"RESULT: Latency={avg_latency*1000:.2f}ms, TPS={tps:.2f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--context", type=int, default=4096)
    args = parser.parse_args()
    
    benchmark_qwen_9b_fp8("models/Qwen2.5-7B-Instruct-GGUF", args.bs, args.context)
