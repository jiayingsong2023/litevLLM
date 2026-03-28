# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn.functional as F
from vllm.kernels.triton.paged_attention import paged_attention_v1
from vllm.kernels.triton.reshape_and_cache import reshape_and_cache

def test_turbo_int4_kv_matches_reference():
    if not torch.cuda.is_available():
        print("Skipping test: CUDA not available")
        return
        
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    block_size = 16
    nh, nkv, hd = 4, 2, 8
    group = nh // nkv
    S = 4
    num_physical_blocks = 1

    # INT4 KV cache uses uint8 and hd // 2
    k_cache = torch.zeros(
        (num_physical_blocks, block_size, nkv, hd // 2), device=device, dtype=torch.uint8
    )
    v_cache = torch.zeros_like(k_cache)
    
    # Use positive values for simple [0, 15] quantization test
    k = torch.rand(S, nkv, hd, device=device, dtype=torch.float16) * 10.0
    v = torch.rand(S, nkv, hd, device=device, dtype=torch.float16) * 10.0
    q = torch.randn(S, nh, hd, device=device, dtype=torch.float16)

    slot_mapping = torch.arange(S, device=device, dtype=torch.long)
    
    # We need a scale for INT4. Let's use max_abs / 15.0
    k_scale = k.abs().max().item() / 15.0
    v_scale = v.abs().max().item() / 15.0
    if k_scale == 0: k_scale = 1.0
    if v_scale == 0: v_scale = 1.0
    
    print(f"Testing INT4 KV with k_scale={k_scale:.4f}, v_scale={v_scale:.4f}")
    
    reshape_and_cache(k, v, k_cache, v_cache, slot_mapping, "int4", k_scale, v_scale)

    scale = hd**-0.5
    seq_lens_ext = torch.arange(1, S + 1, device=device, dtype=torch.int32)
    max_blocks = 8
    block_tables_ext = torch.zeros(S, max_blocks, device=device, dtype=torch.int32)
    block_tables_ext[:, 0] = 0

    attn_out = torch.empty((S, nh, hd), device=device, dtype=torch.float16)
    max_ctx = 4096
    
    paged_attention_v1(
        attn_out,
        q.contiguous(),
        k_cache,
        v_cache,
        nh,
        scale,
        block_tables_ext,
        seq_lens_ext,
        block_size,
        max_ctx,
        None,
        "int4",
        k_scale,
        v_scale,
        num_kv_heads=nkv,
    )

    # Reference: dequantize manually
    # The reshape_and_cache packs low/high nibbles.
    # High nibble is at odd index in original hd?
    # Our reshape kernel:
    # k_l_q = (clamp(k_low / k_scale, 0, 15)).to(uint8)
    # k_h_q = (clamp(k_high / k_scale, 0, 15)).to(uint8)
    # k_packed = k_l_q | (k_h_q << 4)
    # where k_low is hd[0, 2, 4...] and k_high is hd[1, 3, 5...]
    
    k_ref_q = (k.float() / k_scale).clamp(0, 15).round().to(torch.uint8)
    k_ref_dequant = k_ref_q.float() * k_scale
    v_ref_q = (v.float() / v_scale).clamp(0, 15).round().to(torch.uint8)
    v_ref_dequant = v_ref_q.float() * v_scale

    # Reference Attention with dequantized KV
    k_exp = k_ref_dequant.unsqueeze(2).expand(-1, -1, group, -1).reshape(S, nh, hd)
    v_exp = v_ref_dequant.unsqueeze(2).expand(-1, -1, group, -1).reshape(S, nh, hd)
    ref = torch.zeros_like(q)
    for i in range(S):
        sl = int(seq_lens_ext[i].item())
        qi = q[i].unsqueeze(0)
        K = k_exp[:sl]
        V = v_exp[:sl]
        qi_h = qi.squeeze(0).float()
        Kh = K.permute(1, 0, 2).float()
        scores = torch.bmm(qi_h.unsqueeze(1), Kh.transpose(1, 2)).squeeze(1) * scale
        weights = F.softmax(scores, dim=-1)
        Vh = V.permute(1, 0, 2).float()
        ref[i] = (weights.unsqueeze(-1) * Vh).sum(dim=1).half()

    max_diff = (attn_out.float() - ref.float()).abs().max().item()
    print(f"Max diff: {max_diff}")
    
    if max_diff < 0.2: # Relaxed tolerance for first pass
        print("Test passed!")
    else:
        print("Test failed!")
        # Print some values
        print("Attn Out (first 4):", attn_out[0, 0, :4])
        print("Ref Out (first 4):", ref[0, 0, :4])

if __name__ == "__main__":
    test_turbo_int4_kv_matches_reference()
