# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from vllm.kernels.triton.paged_attention import paged_attention_v1


class TestKVSelectiveAttention:
    @pytest.mark.parametrize("select_ratio", [0.0, 0.25, 0.5, 1.0])
    @pytest.mark.parametrize("seq_len", [128, 512])
    def test_selective_vs_full_cosine_sim(self, select_ratio, seq_len):
        """Selective attention output should be cosine-similar to full when ratio=1.0,
        and reasonably close for lower ratios (correctness vs performance trade)."""
        torch.manual_seed(123)
        num_seqs, num_heads, num_kv_heads, head_dim = 1, 8, 4, 128
        block_size = 16
        num_blocks = (seq_len + block_size - 1) // block_size

        q = torch.randn(
            num_seqs, num_heads, head_dim, dtype=torch.float16, device="cuda"
        )
        k_cache = torch.randn(
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            dtype=torch.float16,
            device="cuda",
        )
        v_cache = torch.randn(
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            dtype=torch.float16,
            device="cuda",
        )
        block_tables = torch.arange(
            num_blocks, device="cuda", dtype=torch.int32
        ).unsqueeze(0)
        seq_lens = torch.tensor([seq_len], device="cuda", dtype=torch.int32)

        out_full_triton = torch.empty_like(q)
        paged_attention_v1(
            out_full_triton,
            q,
            k_cache,
            v_cache,
            num_heads,
            1.0,
            block_tables,
            seq_lens,
            block_size,
            seq_len,
            None,
            "fp16",
            kv_select_ratio=0.0,
        )
        torch.cuda.synchronize()

        out_selective_triton = torch.empty_like(q)
        paged_attention_v1(
            out_selective_triton,
            q,
            k_cache,
            v_cache,
            num_heads,
            1.0,
            block_tables,
            seq_lens,
            block_size,
            seq_len,
            None,
            "fp16",
            kv_select_ratio=select_ratio,
            kv_select_min_blocks=2,
        )
        torch.cuda.synchronize()

        cos_sim = torch.nn.functional.cosine_similarity(
            out_full_triton.float().reshape(-1),
            out_selective_triton.float().reshape(-1),
            dim=0,
        )

        if select_ratio in (0.0, 1.0):
            # ratio=0.0 → USE_SELECTION=False (full attention)
            # ratio=1.0 → stride=1 (all blocks loaded)
            assert (
                cos_sim > 0.999
            ), f"ratio={select_ratio} should be exact, got cos_sim={cos_sim:.6f}"
        elif select_ratio >= 0.5:
            # Uniform strided selection with random KV; ~50-75% blocks loaded.
            assert (
                cos_sim > 0.2
            ), f"ratio={select_ratio} cos_sim={cos_sim:.6f} too low"
        else:
            # Very sparse selection; output may differ significantly but
            # must be well-formed (not NaN, not zero).
            assert (
                cos_sim > 0.05
            ), f"ratio={select_ratio} cos_sim={cos_sim:.6f} too low"
            assert not torch.isnan(out_selective_triton).any()
            assert out_selective_triton.abs().max() > 0
