# SPDX-License-Identifier: Apache-2.0
import torch
from vllm.kernels.triton.reshape_and_cache import reshape_and_cache

def test_multi_turn_simulation():
    print("--- LitevLLM Multi-Turn Cache Simulation ---")
    
    # 配置参数
    num_heads = 8
    head_size = 64
    block_size = 16  # 每个物理块 16 个 token
    num_blocks = 4   # 总共 4 个物理块
    
    # 1. 初始化物理 KV Cache (Paged Format)
    # 形状: [num_blocks, block_size, num_heads, head_size]
    key_cache = torch.zeros((num_blocks, block_size, num_heads, head_size), device="cuda", dtype=torch.float16)
    value_cache = torch.zeros((num_blocks, block_size, num_heads, head_size), device="cuda", dtype=torch.float16)
    
    # --- 第一轮对话 (Prefill: 20 tokens) ---
    print("\n[Turn 1] Prefilling 20 tokens...")
    num_tokens_t1 = 20
    k_t1 = torch.randn((num_tokens_t1, num_heads, head_size), device="cuda", dtype=torch.float16)
    v_t1 = torch.randn((num_tokens_t1, num_heads, head_size), device="cuda", dtype=torch.float16)
    
    # 模拟调度器分配的槽位: 前 16 个在 block 0, 后 4 个在 block 1
    slot_mapping_t1 = torch.arange(0, num_tokens_t1, device="cuda", dtype=torch.int32)
    
    # 执行写入
    reshape_and_cache(k_t1, v_t1, key_cache, value_cache, slot_mapping_t1, "float16")
    
    # 验证第一轮数据
    # 检查 Block 0 是否填满
    assert torch.allclose(key_cache[0], k_t1[:16], atol=1e-3)
    # 检查 Block 1 的前 4 个位置
    assert torch.allclose(key_cache[1, :4], k_t1[16:], atol=1e-3)
    print("Turn 1 Verification: SUCCESS (Paged logic works)")

    # --- 第二轮对话 (Incremental: 5 new tokens) ---
    print("\n[Turn 2] Adding 5 new tokens...")
    num_tokens_t2 = 5
    k_t2 = torch.randn((num_tokens_t2, num_heads, head_size), device="cuda", dtype=torch.float16)
    v_t2 = torch.randn((num_tokens_t2, num_heads, head_size), device="cuda", dtype=torch.float16)
    
    # 槽位从 20 开始: 都在 block 1 (位置 4 到 8)
    slot_mapping_t2 = torch.arange(num_tokens_t1, num_tokens_t1 + num_tokens_t2, device="cuda", dtype=torch.int32)
    
    # 执行增量写入
    reshape_and_cache(k_t2, v_t2, key_cache, value_cache, slot_mapping_t2, "float16")
    
    # 验证第二轮数据
    # 检查 Block 1 的新位置 [4:9]
    assert torch.allclose(key_cache[1, 4:9], k_t2, atol=1e-3)
    # 检查 Block 1 的旧位置 [0:4] 是否被破坏
    assert torch.allclose(key_cache[1, :4], k_t1[16:], atol=1e-3)
    
    print("Turn 2 Verification: SUCCESS (Incremental update works)")
    print("\nFinal Status: All cache consistency checks PASSED.")

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_multi_turn_simulation()
    else:
        print("CUDA not available, skipping test.")
