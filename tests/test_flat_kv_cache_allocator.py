# SPDX-License-Identifier: Apache-2.0
import torch

from vllm.engine.initialization.flat_kv_cache_allocator import FlatKVCacheAllocator


def test_flat_allocator_uniform_layers():
    allocator = FlatKVCacheAllocator(
        num_layers=4,
        num_total_blocks=32,
        block_size=16,
        device=torch.device("cpu"),
    )
    kv_caches, kv_scale_caches, returned_num_blocks = allocator.allocate(
        layer_kv_specs=None,
        kv_dtype=torch.float16,
        kv_head_dim=64,
        fallback_num_kv_heads=8,
        fallback_kv_head_dim=64,
        needs_scale_cache=False,
    )
    assert returned_num_blocks == 32
    assert len(kv_caches) == 4
    for k, v in kv_caches:
        assert k.shape == (32, 16, 8, 64)
        assert v.shape == k.shape
        assert k.is_contiguous()
        assert k.storage().data_ptr() == v.storage().data_ptr()


def test_flat_allocator_shared_storage_across_layers():
    allocator = FlatKVCacheAllocator(
        num_layers=2,
        num_total_blocks=16,
        block_size=8,
        device=torch.device("cpu"),
    )
    kv_caches, _, _ = allocator.allocate(
        layer_kv_specs=None,
        kv_dtype=torch.float16,
        kv_head_dim=32,
        fallback_num_kv_heads=4,
        fallback_kv_head_dim=32,
        needs_scale_cache=False,
    )
    ptrs = [k.untyped_storage().data_ptr() for k, _ in kv_caches]
    ptrs += [v.untyped_storage().data_ptr() for _, v in kv_caches]
    assert len(set(ptrs)) == 1


def test_flat_allocator_per_layer_specs_and_scales():
    allocator = FlatKVCacheAllocator(
        num_layers=2,
        num_total_blocks=8,
        block_size=16,
        device=torch.device("cpu"),
    )
    layer_kv_specs = [(4, 64), (2, 128)]
    kv_caches, kv_scale_caches, _ = allocator.allocate(
        layer_kv_specs=layer_kv_specs,
        kv_dtype=torch.uint8,
        kv_head_dim=64,
        fallback_num_kv_heads=4,
        fallback_kv_head_dim=64,
        needs_scale_cache=True,
    )
    assert kv_caches[0][0].shape == (8, 16, 4, 32)  # uint8 packed halves head_dim
    assert kv_caches[1][0].shape == (8, 16, 2, 64)
    assert kv_scale_caches[0][0].shape == (8, 16, 4, 1)
    assert kv_scale_caches[1][0].shape == (8, 16, 2, 1)

    expected_scale_elems = 2 * sum(8 * 16 * nkv for nkv, _ in layer_kv_specs)
    scale_storage_elems = (
        kv_scale_caches[0][0].untyped_storage().size() // torch.float32.itemsize
    )
    assert scale_storage_elems == expected_scale_elems
