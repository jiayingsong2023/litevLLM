import torch
import triton
import triton.language as tl

@triton.jit
def _paged_attention_kernel(
    Out_ptr,             # [num_seqs, num_heads, head_size]
    Q_ptr,               # [num_seqs, num_heads, head_size]
    K_ptr,               # [num_blocks, num_kv_heads, head_size/x, block_size, x]
    V_ptr,               # [num_blocks, num_kv_heads, head_size, block_size]
    BlockTables_ptr,     # [num_seqs, max_num_blocks_per_seq]
    SeqLens_ptr,         # [num_seqs]
    Alibi_ptr,           # Optional [num_heads, max_seq_len]
    
    scale,               # float
    num_seqs, 
    num_heads, 
    num_kv_heads, 
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks_per_seq,
    
    stride_out_seq, stride_out_head, stride_out_dim,
    stride_q_seq, stride_q_head, stride_q_dim,
    stride_k_block, stride_k_head, stride_k_dim, stride_k_token, stride_k_x,
    stride_v_block, stride_v_head, stride_v_dim, stride_v_token,
    stride_bt_seq, stride_bt_block,
    stride_sl_seq,
    
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KV_X: tl.constexpr, # The 'x' dimension size
):
    pid = tl.program_id(0)
    
    seq_idx = pid // num_heads
    head_idx = pid % num_heads
    kv_head_idx = head_idx // (num_heads // num_kv_heads)
    
    seq_len = tl.load(SeqLens_ptr + seq_idx * stride_sl_seq)
    
    # Load Query
    off_q = seq_idx * stride_q_seq + head_idx * stride_q_head + tl.arange(0, BLOCK_D) * stride_q_dim
    q = tl.load(Q_ptr + off_q)
    
    m_i = -float('inf')
    l_i = 1.0 
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    
    qk_scale = scale
    
    num_blocks = (seq_len + block_size - 1) // block_size
    
    # Loop over blocks
    for i in range(0, num_blocks):
        block_idx_ptr = BlockTables_ptr + seq_idx * stride_bt_seq + i * stride_bt_block
        physical_block_idx = tl.load(block_idx_ptr)
        
        offs_n = tl.arange(0, BLOCK_N)
        global_token_idx = i * block_size + offs_n
        block_mask = global_token_idx < seq_len
        
        # Load K
        # Logic handles both 4D (KV_X=1) and 5D (KV_X>1) layouts provided strides are correct.
        
        offs_d = tl.arange(0, BLOCK_D)
        
        offs_n_2d = offs_n[:, None] # [N, 1]
        offs_d_2d = offs_d[None, :] # [1, D]
        
        idx_d_outer = offs_d_2d // KV_X
        idx_d_inner = offs_d_2d % KV_X
        
        off_k_base = physical_block_idx * stride_k_block + kv_head_idx * stride_k_head
        off_k = off_k_base + idx_d_outer * stride_k_dim + offs_n_2d * stride_k_token + idx_d_inner * stride_k_x
        
        k = tl.load(K_ptr + off_k, mask=block_mask[:, None], other=0.0)
        
        # Compute QK
        qk = tl.sum(q[None, :] * k, axis=1) # [N]
        qk *= qk_scale
        
        # Update Softmax statistics
        qk_masked = tl.where(block_mask, qk, -float('inf'))
        m_curr = tl.max(qk_masked, axis=0)
        m_new = tl.maximum(m_i, m_curr)
        
        alpha = tl.exp(m_i - m_new)
        exp_qk = tl.exp(qk_masked - m_new)
        
        l_curr = tl.sum(exp_qk, axis=0)
        l_new = alpha * l_i + l_curr
        
        # Load V
        off_v_base = physical_block_idx * stride_v_block + kv_head_idx * stride_v_head
        off_v = off_v_base + offs_d_2d * stride_v_dim + offs_n_2d * stride_v_token
        
        v = tl.load(V_ptr + off_v, mask=block_mask[:, None], other=0.0)
        
        # Weighted sum
        w_v = tl.sum(exp_qk[:, None] * v, axis=0) # [D]
        
        acc = acc * alpha + w_v
        
        l_i = l_new
        m_i = m_new

    # Finalize
    out = acc / l_i
    
    off_out = seq_idx * stride_out_seq + head_idx * stride_out_head + tl.arange(0, BLOCK_D) * stride_out_dim
    tl.store(Out_ptr + off_out, out)

def paged_attention_v1(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: torch.Tensor | None,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
) -> None:
    num_seqs, num_heads, head_size = query.shape
    max_num_blocks_per_seq = block_tables.shape[1]
    
    s_k = key_cache.stride()
    if len(s_k) == 5:
        stride_k_block, stride_k_head, stride_k_dim, stride_k_token, stride_k_x = s_k
        kv_x = key_cache.shape[-1]
    else:
        # Fallback 4D: [blocks, heads, head_size, block_size]
        # Treat x=1
        stride_k_block, stride_k_head, stride_k_dim, stride_k_token = s_k
        stride_k_x = 1
        kv_x = 1
        
    s_v = value_cache.stride()
    if len(s_v) == 4:
        stride_v_block, stride_v_head, stride_v_dim, stride_v_token = s_v
    else:
        stride_v_block, stride_v_head, stride_v_dim, stride_v_token = s_v
        
    stride_bt_seq, stride_bt_block = block_tables.stride()
    stride_sl_seq = seq_lens.stride(0)

    grid = (num_seqs * num_heads,)
    
    _paged_attention_kernel[grid](
        out, query, key_cache, value_cache, block_tables, seq_lens, alibi_slopes,
        scale, num_seqs, num_heads, num_kv_heads, head_size, block_size, max_num_blocks_per_seq,
        stride_out_seq=out.stride(0), stride_out_head=out.stride(1), stride_out_dim=out.stride(2),
        stride_q_seq=query.stride(0), stride_q_head=query.stride(1), stride_q_dim=query.stride(2),
        stride_k_block=stride_k_block, stride_k_head=stride_k_head, stride_k_dim=stride_k_dim, stride_k_token=stride_k_token, stride_k_x=stride_k_x,
        stride_v_block=stride_v_block, stride_v_head=stride_v_head, stride_v_dim=stride_v_dim, stride_v_token=stride_v_token,
        stride_bt_seq=stride_bt_seq, stride_bt_block=stride_bt_block,
        stride_sl_seq=stride_sl_seq,
        BLOCK_D=head_size,
        BLOCK_N=block_size,
        KV_X=kv_x,
    )

def paged_attention_v2(
    out: torch.Tensor,
    exp_sum: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: torch.Tensor | None,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
) -> None:
    paged_attention_v1(
        out, query, key_cache, value_cache, num_kv_heads, scale, block_tables,
        seq_lens, block_size, max_seq_len, alibi_slopes, kv_cache_dtype,
        k_scale, v_scale, tp_rank, blocksparse_local_blocks, 
        blocksparse_vert_stride, blocksparse_block_size, blocksparse_head_sliding_step
    )