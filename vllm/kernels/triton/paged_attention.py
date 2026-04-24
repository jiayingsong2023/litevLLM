# SPDX-License-Identifier: Apache-2.0
import os

import torch
import triton
import triton.language as tl


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(raw.strip())
    except Exception:
        return int(default)


def _env_truthy(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _select_paged_attention_launch_config(
    *,
    num_seqs: int,
    head_size: int,
    block_size: int,
    is_int4: bool,
    is_fp8: bool,
    attn_scope: str | None = None,
    layer_type: str | None = None,
) -> tuple[int, int]:
    scope = str(attn_scope or "").strip().lower()
    if not scope:
        lt = str(layer_type or "").strip().lower()
        if "sliding" in lt or "local" in lt:
            scope = "local"
        elif "full" in lt or "global" in lt:
            scope = "global"

    # Scope-specific override: let local/global layers use independent launch knobs.
    # Alias full->global and sliding->local for convenience.
    if scope in ("full",):
        scope = "global"
    if scope in ("sliding",):
        scope = "local"
    if scope in ("local", "global"):
        scoped_warps = _env_int(
            f"FASTINFERENCE_PAGED_ATTN_NUM_WARPS_{scope.upper()}",
            -1,
        )
        scoped_stages = _env_int(
            f"FASTINFERENCE_PAGED_ATTN_NUM_STAGES_{scope.upper()}",
            -1,
        )
        if scoped_warps > 0 and scoped_stages > 0:
            return scoped_warps, scoped_stages

    # C1 single-request preset for consumer GPUs / Ollama-like workloads.
    # Keep this ahead of generic/manual-tuning heuristics and behind explicit
    # scope overrides so operators can still force an emergency setting.
    if _env_truthy("FASTINFERENCE_GEMMA4_C1_PRESET") and int(num_seqs) <= 1:
        if scope == "global":
            return 4, 1
        if scope == "local":
            return 4, 1

    # Manual override is useful for A/B and emergency rollback without code edit.
    override_warps = _env_int("FASTINFERENCE_PAGED_ATTN_NUM_WARPS", -1)
    override_stages = _env_int("FASTINFERENCE_PAGED_ATTN_NUM_STAGES", -1)
    if override_warps > 0 and override_stages > 0:
        return override_warps, override_stages

    # Decode-heavy Gemma4-31B path: head_size=256, block_size=16.
    # Scope-aware bucket:
    # - global/full attention: low concurrency often benefits from leaner launch (2/2),
    #   while >=3 concurrency is more stable with 4/2.
    # - local/sliding attention: use a lighter stage depth at c1 (4/1), and 4/2 otherwise.
    # Fallback keeps stable default and env override remains available.
    if head_size >= 256:
        if scope == "global":
            if num_seqs <= 2:
                return 2, 2
            return 4, 2
        if scope == "local":
            if num_seqs <= 1:
                return 4, 1
            return 4, 2
        return 4, 2

    if is_int4 or is_fp8:
        if num_seqs <= 8:
            return 4, 2
        return 2, 2

    if block_size >= 32:
        return 4, 2
    return 2, 2

@triton.jit
def _paged_attention_kernel(
    Out_ptr, Q_ptr, K_ptr, V_ptr, 
    BlockTables_ptr,
    SeqLens_ptr,
    K_Ptrs_ptr, V_Ptrs_ptr,
    scale, num_seqs, num_heads, num_kv_heads, head_size: tl.constexpr,
    block_size: tl.constexpr, max_num_blocks_per_seq,
    stride_out_seq, stride_out_head, stride_out_dim,
    stride_q_seq, stride_q_head, stride_q_dim,
    stride_k_block, stride_k_token, stride_k_head, stride_k_dim,
    stride_v_block, stride_v_token, stride_v_head, stride_v_dim,
    stride_bt_seq, stride_bt_block, stride_sl_seq,
    k_scale_arg, v_scale_arg,
    K_Scale_Ptrs_ptr, V_Scale_Ptrs_ptr,
    stride_ks_block, stride_ks_token, stride_ks_head,
    stride_vs_block, stride_vs_token, stride_vs_head,
    IS_FP8: tl.constexpr,
    IS_INT4: tl.constexpr,
    IS_CACHED: tl.constexpr,
    HAS_ROW_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    seq_idx = pid // num_heads; head_idx = pid % num_heads
    kv_head_idx = head_idx // (num_heads // num_kv_heads)
    seq_len = tl.load(SeqLens_ptr + seq_idx * stride_sl_seq)
    
    # Initialize accumulators
    m_i = -float('inf'); l_i = 0.0
    acc_low = tl.zeros([BLOCK_D // 2], dtype=tl.float32)
    acc_high = tl.zeros([BLOCK_D // 2], dtype=tl.float32)
    acc_full = tl.zeros([BLOCK_D], dtype=tl.float32)

    if IS_INT4:
        offs_d_half = tl.arange(0, BLOCK_D // 2)
        offs_d_half_2d = offs_d_half[None, :]
        # Contiguous layout: first half of channels in low nibbles, second half in high nibbles
        q_low = tl.load(Q_ptr + seq_idx * stride_q_seq + head_idx * stride_q_head + offs_d_half * stride_q_dim).to(tl.float32)
        q_high = tl.load(Q_ptr + seq_idx * stride_q_seq + head_idx * stride_q_head + (offs_d_half + BLOCK_D // 2) * stride_q_dim).to(tl.float32)
    else:
        offs_d_2d = tl.arange(0, BLOCK_D)[None, :]
        off_q = seq_idx * stride_q_seq + head_idx * stride_q_head + tl.arange(0, BLOCK_D) * stride_q_dim
        q = tl.load(Q_ptr + off_q).to(tl.float32)

    num_blocks = (seq_len + block_size - 1) // block_size
    offs_n = tl.arange(0, BLOCK_N)
    
    for i in range(0, num_blocks):
        if IS_CACHED:
            k_base_ptr = tl.load(K_Ptrs_ptr + seq_idx * max_num_blocks_per_seq + i).to(tl.pointer_type(tl.uint8 if IS_INT4 else tl.float16))
            v_base_ptr = tl.load(V_Ptrs_ptr + seq_idx * max_num_blocks_per_seq + i).to(tl.pointer_type(tl.uint8 if IS_INT4 else tl.float16))
            if HAS_ROW_SCALE:
                ks_base_ptr = tl.load(K_Scale_Ptrs_ptr + seq_idx * max_num_blocks_per_seq + i).to(tl.pointer_type(tl.float32))
                vs_base_ptr = tl.load(V_Scale_Ptrs_ptr + seq_idx * max_num_blocks_per_seq + i).to(tl.pointer_type(tl.float32))
        else:
            block_idx = tl.load(BlockTables_ptr + seq_idx * stride_bt_seq + i * stride_bt_block)
            k_base_ptr = K_ptr + block_idx * stride_k_block
            v_base_ptr = V_ptr + block_idx * stride_v_block
            if HAS_ROW_SCALE:
                ks_base_ptr = K_Scale_Ptrs_ptr + block_idx * stride_ks_block
                vs_base_ptr = V_Scale_Ptrs_ptr + block_idx * stride_vs_block

        global_token_idx = i * block_size + offs_n
        block_mask = global_token_idx < seq_len
        
        k_scale = k_scale_arg
        v_scale = v_scale_arg
        if HAS_ROW_SCALE:
            k_scale = tl.load(ks_base_ptr + offs_n * stride_ks_token + kv_head_idx * stride_ks_head, mask=block_mask, other=1.0)
            v_scale = tl.load(vs_base_ptr + offs_n * stride_vs_token + kv_head_idx * stride_vs_head, mask=block_mask, other=1.0)
            k_scale = k_scale[:, None]
            v_scale = v_scale[:, None]

        if IS_INT4:
            off_kv = offs_n[:, None] * stride_k_token + kv_head_idx * stride_k_head + offs_d_half_2d * stride_k_dim
            k_packed = tl.load(k_base_ptr.to(tl.pointer_type(tl.uint8)) + off_kv, mask=block_mask[:, None], other=0).to(tl.int32)
            # Manual sign-extension for 4-bit signed
            k_l = ((k_packed << 28) >> 28).to(tl.float32) * k_scale
            k_h = ((k_packed << 24) >> 28).to(tl.float32) * k_scale
            qk = tl.sum(q_low[None, :] * k_l + q_high[None, :] * k_h, axis=1) * scale
        else:
            off_kv = offs_n[:, None] * stride_k_token + kv_head_idx * stride_k_head + offs_d_2d * stride_k_dim
            k = tl.load(k_base_ptr + off_kv, mask=block_mask[:, None], other=0.0)
            if IS_FP8: k = (k.to(tl.float32) * k_scale)
            qk = tl.sum(q[None, :] * k, axis=1) * scale

        qk = tl.where(block_mask, qk, -float('inf'))
        m_curr = tl.max(qk, axis=0); m_new = tl.maximum(m_i, m_curr)
        
        # Guard against -inf - (-inf) to avoid NaN
        delta_m = tl.where(m_i == -float('inf'), 0.0, m_i - m_new)
        alpha = tl.exp(delta_m).to(tl.float32)
        
        p = tl.exp((qk - m_new).to(tl.float32)).to(tl.float32)
        p = tl.where(block_mask, p, 0.0).to(tl.float32)
        l_curr = tl.sum(p, axis=0).to(tl.float32)
        l_new = (alpha * l_i + l_curr).to(tl.float32)
        
        # Guard against zero sum to avoid NaN in division
        l_new = tl.maximum(l_new, 1e-20)
        
        if IS_INT4:
            v_packed = tl.load(v_base_ptr.to(tl.pointer_type(tl.uint8)) + offs_n[:, None] * stride_v_token + kv_head_idx * stride_v_head + offs_d_half_2d * stride_v_dim, mask=block_mask[:, None], other=0).to(tl.int32)
            # Manual sign-extension for 4-bit signed
            v_l = ((v_packed << 28) >> 28).to(tl.float32) * v_scale
            v_h = ((v_packed << 24) >> 28).to(tl.float32) * v_scale
            acc_low = acc_low * alpha + tl.sum(p[:, None] * v_l, axis=0)
            acc_high = acc_high * alpha + tl.sum(p[:, None] * v_h, axis=0)
        else:
            v = tl.load(v_base_ptr + offs_n[:, None] * stride_v_token + kv_head_idx * stride_v_head + offs_d_2d * stride_v_dim, mask=block_mask[:, None], other=0.0)
            if IS_FP8: v = (v.to(tl.float32) * v_scale)
            acc_full = acc_full * alpha + tl.sum(p[:, None] * v, axis=0)
        
        l_i = l_new; m_i = m_new

    off_out_base = seq_idx * stride_out_seq + head_idx * stride_out_head
    inv_l_i = 1.0 / tl.maximum(l_i, 1e-20)
    if IS_INT4:
        # Standard Online Softmax normalization in float32
        res_low = (acc_low * inv_l_i)
        res_high = (acc_high * inv_l_i)
        off_d_half = tl.arange(0, BLOCK_D // 2)
        tl.store(Out_ptr + off_out_base + off_d_half * stride_out_dim, res_low.to(Out_ptr.dtype.element_ty))
        tl.store(Out_ptr + off_out_base + (off_d_half + BLOCK_D // 2) * stride_out_dim, res_high.to(Out_ptr.dtype.element_ty))
    else:
        out = (acc_full * inv_l_i)
        off_d = tl.arange(0, BLOCK_D)
        tl.store(Out_ptr + off_out_base + off_d * stride_out_dim, out.to(Out_ptr.dtype.element_ty))

def paged_attention_v1(out, query, key_cache, value_cache, num_heads, scale, block_tables, seq_lens, block_size, max_seq_len, alibi_slopes, kv_cache_dtype, k_scale=1.0, v_scale=1.0, k_ptrs=None, v_ptrs=None, k_scale_ptrs=None, v_scale_ptrs=None, **kwargs):
    num_seqs, _num_heads_check, head_size = query.shape
    num_kv_heads = kwargs.get("num_kv_heads", num_heads)
    is_fp8 = "fp8" in str(kv_cache_dtype).lower()
    is_int4 = "int4" in str(kv_cache_dtype).lower()
    is_cached = k_ptrs is not None
    has_row_scale = k_scale_ptrs is not None
    
    s_k = key_cache.stride(); s_v = value_cache.stride()
    
    if has_row_scale:
        s_ks = k_scale_ptrs.stride(); s_vs = v_scale_ptrs.stride()
    else:
        s_ks = [0]*4; s_vs = [0]*4

    num_warps, num_stages = _select_paged_attention_launch_config(
        num_seqs=num_seqs,
        head_size=head_size,
        block_size=int(block_size),
        is_int4=is_int4,
        is_fp8=is_fp8,
        attn_scope=kwargs.get("attn_scope"),
        layer_type=kwargs.get("layer_type"),
    )

    grid = (num_seqs * num_heads,)
    _paged_attention_kernel[grid](
        out, query, key_cache, value_cache, 
        block_tables if not is_cached else out, seq_lens,
        k_ptrs if is_cached else out, v_ptrs if is_cached else out,
        scale, num_seqs, num_heads, num_kv_heads, head_size, block_size, block_tables.shape[1] if block_tables is not None else 0,
        out.stride(0), out.stride(1), out.stride(2),
        query.stride(0), query.stride(1), query.stride(2),
        s_k[0], s_k[1], s_k[2], s_k[3],
        s_v[0], s_v[1], s_v[2], s_v[3],
        block_tables.stride(0) if block_tables is not None else 0,
        block_tables.stride(1) if block_tables is not None else 0,
        seq_lens.stride(0),
        k_scale if not has_row_scale else 1.0, v_scale if not has_row_scale else 1.0,
        k_scale_ptrs if has_row_scale else out, v_scale_ptrs if has_row_scale else out,
        s_ks[0], s_ks[1], s_ks[2],
        s_vs[0], s_vs[1], s_vs[2],
        IS_FP8=is_fp8, IS_INT4=is_int4, IS_CACHED=is_cached, HAS_ROW_SCALE=has_row_scale, BLOCK_D=head_size, BLOCK_N=block_size,
        num_warps=num_warps, num_stages=num_stages,
    )

def paged_attention_v2(*args, **kwargs):
    return paged_attention_v1(*args, **kwargs)
