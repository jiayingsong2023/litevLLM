# SPDX-License-Identifier: Apache-2.0
import torch

from vllm.kernels.triton.paged_attention import paged_attention_v1


def triton_paged_attention(
    q, k, v, kv_cache, slot_mapping, seq_lens, block_tables, scale, **kwargs
):
    """
    LitevLLM Triton Paged Attention Wrapper.
    Passes K/V scales and handles INT4/FP8 types.
    """
    # 1. Prepare scales (if not provided, check the forward context or layer attributes)
    # Note: In a real vLLM implementation, these would come from the Attention layer.
    # Here we default to 1.0 but expect them in kwargs if quantization is active.
    config = kwargs.get("config")
    k_scale = getattr(config, "k_scale", kwargs.get("k_scale", 1.0))
    v_scale = getattr(config, "v_scale", kwargs.get("v_scale", 1.0))
    kv_cache_dtype = getattr(
        config, "kv_type", kwargs.get("kv_cache_dtype", "turbo_int4")
    )

    # 2. Extract Key/Value Cache (Standard vLLM layout)
    # kv_cache[0] is Key, kv_cache[1] is Value
    if isinstance(kv_cache, (list, tuple)):
        key_cache, value_cache = kv_cache[0], kv_cache[1]
    else:
        # If it's a single tensor, it might be interleaved or we need to split
        # but standard paged attention uses a list of two tensors.
        key_cache, value_cache = kv_cache, kv_cache

    # 3. Output buffer
    out = torch.empty_like(q)
    num_heads = q.shape[1]
    block_size = key_cache.shape[1]
    max_seq_len = 0

    # 4. Invoke Triton Kernel
    paged_attention_v1(
        out,
        q,
        key_cache,
        value_cache,
        num_heads,
        scale,
        block_tables,
        seq_lens,
        block_size,
        max_seq_len,
        None,
        kv_cache_dtype,
        k_scale=k_scale,
        v_scale=v_scale,
        config=config,
    )

    return out
