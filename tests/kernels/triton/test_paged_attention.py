
import torch
import pytest
from vllm.kernels.triton.paged_attention import paged_attention_v1
from vllm.config import VllmConfig
from vllm.config.vllm import set_current_vllm_config

def naive_paged_attention(
    out,
    query,
    key_cache,
    value_cache,
    num_kv_heads,
    scale,
    block_tables,
    seq_lens,
    block_size,
):
    num_seqs, num_heads, head_size = query.shape
    
    for i in range(num_seqs):
        sl = seq_lens[i].item()
        for h in range(num_heads):
            q = query[i, h] # [D]
            kv_h = h // (num_heads // num_kv_heads)
            
            # Gather K, V
            keys = []
            values = []
            
            # Block indices
            num_blocks = (sl + block_size - 1) // block_size
            for b in range(num_blocks):
                block_idx = block_tables[i, b].item()
                # K shape: [blocks, kv_heads, head_size/x, block_size, x]
                # We need to reconstruct [block_size, head_size]
                # Assuming standard layout where we can just flatten and reshape?
                # Actually, let's just use the same indexing logic as the kernel or assume 4D simplified for test?
                # To make test easier, let's use 4D K cache [blocks, kv_heads, head_size, block_size]
                # checking if kernel handles it (kernel has fallback logic).
                
                k_block = key_cache[block_idx, kv_h] # [head_size, block_size]
                v_block = value_cache[block_idx, kv_h] # [head_size, block_size]
                
                # Transpose to [block_size, head_size]
                keys.append(k_block.t())
                values.append(v_block.t())
                
            K = torch.cat(keys, dim=0) # [total_blocks * block_size, D]
            V = torch.cat(values, dim=0)
            
            # Trim to seq_len
            K = K[:sl]
            V = V[:sl]
            
            # Attention
            scores = torch.matmul(q, K.t()) * scale # [sl]
            probs = torch.softmax(scores, dim=-1)
            output = torch.matmul(probs, V) # [D]
            
            out[i, h] = output

@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("num_heads", [4])
@pytest.mark.parametrize("num_kv_heads", [4, 1]) # MQA
@pytest.mark.parametrize("num_seqs", [2])
def test_paged_attention_correctness(
    block_size,
    head_size,
    num_heads,
    num_kv_heads,
    num_seqs,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    device = "cuda"
    dtype = torch.float16
    
    # Setup dummy config
    config = VllmConfig()
    with set_current_vllm_config(config):
    
        max_seq_len = 128
        max_num_blocks = (max_seq_len + block_size - 1) // block_size
        num_blocks = num_seqs * max_num_blocks + 10 # Pool size
        
        # Init inputs
        query = torch.randn(num_seqs, num_heads, head_size, device=device, dtype=dtype)
        
        # Simplified 4D K Cache for easier verification: [blocks, kv_heads, head_size, block_size]
        key_cache = torch.randn(num_blocks, num_kv_heads, head_size, block_size, device=device, dtype=dtype)
        value_cache = torch.randn(num_blocks, num_kv_heads, head_size, block_size, device=device, dtype=dtype)
        
        block_tables = torch.randint(0, num_blocks, (num_seqs, max_num_blocks), device=device, dtype=torch.int32)
        seq_lens = torch.randint(1, max_seq_len, (num_seqs,), device=device, dtype=torch.int32)
        
        scale = 1.0 / (head_size ** 0.5)
        
        out_triton = torch.zeros_like(query)
        out_ref = torch.zeros_like(query)
        
        # Run Triton
        # Note: We pass None for optional args not used in v1 core logic
        paged_attention_v1(
            out_triton,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            block_tables,
            seq_lens,
            block_size,
            max_seq_len,
            None, # alibi
            "auto", # dtype
            torch.tensor(1.0, device=device), # k_scale
            torch.tensor(1.0, device=device), # v_scale
        )
        
        # Run Reference
        naive_paged_attention(
            out_ref,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            block_tables,
            seq_lens,
            block_size,
        )
        
        # Compare
        diff = (out_triton - out_ref).abs().max()
        print(f"Max Diff: {diff}")
        assert diff < 1e-2 # FP16 precision
