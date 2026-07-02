# KV Flat Pool + Dynamic Block Allocation Design

**Date:** 2026-07-02  
**Author:** Kimi Code CLI  
**Status:** Pending user review

## Summary

The current `LiteEngine` allocates one independent `(K, V)` tensor per layer. Each sequence slot reserves a fixed, contiguous range of `num_blocks_per_seq` blocks. This has two drawbacks:

1. **Allocator overhead / fragmentation:** `num_layers * 2` (plus scales) separate CUDA allocations and per-tensor metadata.
2. **Static per-slot reservation:** a short request consumes the same maximum block budget as a long request, and a single request can never use blocks left idle by other slots.

This design replaces the per-layer allocations with a single flat GPU buffer and assigns physical blocks to requests dynamically from a global free list. It deliberately **does not** introduce upstream v1 features such as block-level prefix caching, COW sharing, or hybrid KV groups — those are left for a future phase.

## Goals

- Allocate all per-layer K/V caches (and scale caches) from one contiguous GPU buffer.
- Maintain per-layer views that keep the existing PagedAttention / `reshape_and_cache` tensor contract unchanged.
- Replace fixed per-slot block ranges with request-scoped dynamic block IDs.
- Keep `PagedAttention` kernels, `reshape_and_cache`, and model-layer code unchanged.
- Preserve existing prefix-cache behavior at the request level (whole-prompt clone/materialize).
- Pass the existing fast and correctness regression suites without behavioral regressions.

## Non-Goals

- Block-level prefix caching or cross-request block sharing (would require v1-style `BlockPool` + `KVCacheManager`).
- Preemption / recomputation logic changes.
- Changing DeepSeek V4 Flash direct-runtime path (it uses its own `DeepSeekV4CompressedKVCache`).
- Changing the KV cache dtype, block size, or quantization scheme.
- Distributed KV cache (DCP/PCP) support beyond what already works today.

## Proposed Architecture

```text
LiteEngine.__init__
  ├─ compute per-layer KV specs
  ├─ FlatKVCacheAllocator.allocate(...) -> (kv_caches, kv_scale_caches, num_total_blocks)
  ├─ BlockAllocator(num_total_blocks)
  └─ LiteRuntimeAssembler.assemble(..., block_allocator=block_allocator, ...)

RuntimeController.run_step
  ├─ step_plan = StepScheduler.build_plan(scheduler)
  ├─ KVBlockManager.ensure_blocks_for_requests(request_ids, token_counts)
  ├─ execute prefill / decode
  └─ KVBlockManager.free_request_blocks(finished_request_ids)
```

### Component 1: `FlatKVCacheAllocator`

**Location:** `vllm/engine/initialization/flat_kv_cache_allocator.py`

Responsibility: allocate one contiguous GPU tensor for all K/V cache data and one for scale caches, then return per-layer views in the shape the existing kernels expect.

#### Layout

For each layer `i`:

```text
block_elems_i = block_size * num_kv_heads_i * kv_head_dim_i
layer_kv_elems_i = num_total_blocks * block_elems_i
```

The flat buffer has:

```text
total_kv_elems = 2 * num_total_blocks * sum_i(block_elems_i)
kv_buffer = torch.zeros(total_kv_elems, dtype=kv_dtype, device=device)
```

The first half is K, the second half is V, each half laid out layer by layer:

```text
K offset for layer i = num_total_blocks * sum_{j<i} block_elems_j
V offset for layer i = half_size + K offset for layer i
```

Each per-layer view is:

```python
k_view = kv_buffer[k_off : k_off + layer_kv_elems_i].view(
    num_total_blocks, block_size, num_kv_heads_i, kv_head_dim_i
)
```

Scale caches use the same pattern in a separate `float32` buffer because they are always `float32` and have shape `(num_total_blocks, block_size, num_kv_heads_i, 1)`.

All views are contiguous slices of the underlying buffer, so strides remain what the Triton kernels expect. The exact internal byte ordering is private to the allocator; callers only receive the per-layer view list.

#### Interface

```python
class FlatKVCacheAllocator:
    def allocate(
        self,
        layer_kv_specs: list[tuple[int, int]] | None,
        kv_dtype: torch.dtype,
        kv_head_dim: int,
        fallback_num_kv_heads: int,
        fallback_kv_head_dim: int,
        needs_scale_cache: bool,
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]],
               list[tuple[torch.Tensor | None, torch.Tensor | None]],
               int]:
        ...
```

The returned `num_total_blocks` is the caller-supplied capacity. Block ID 0 will be reserved as the zeroed null block (see Component 2), so the effective usable count is `num_total_blocks - 1`.

### Component 2: `BlockAllocator`

**Location:** `vllm/engine/block_allocator.py`

Responsibility: manage a global free list of physical block IDs.

#### Null block

Block ID `0` is reserved and never handed out. It is zeroed at startup and used to pad block tables for unallocated positions. This avoids accidental out-of-bounds reads if a kernel ever iterates one block too far, and it leaves a clean path for future COW/prefix sharing.

#### Interface

```python
class BlockAllocator:
    def __init__(self, num_total_blocks: int) -> None: ...

    def allocate(self, n: int) -> list[int]:
        """Return n free block IDs. Raise if not enough blocks."""

    def free(self, block_ids: Iterable[int]) -> None:
        """Return block IDs to the free list."""

    @property
    def num_free(self) -> int: ...

    def can_allocate(self, n: int) -> bool: ...
```

Free list ordering: IDs are popped from the front and appended to the back. For the scope of this design, LRU ordering is not required because there is no cross-request sharing. The order only affects which stale block is reused next.

### Component 3: `KVBlockManager` updates

**Location:** `vllm/engine/kv_block_manager.py`

Responsibility: own the block allocator and maintain per-request block tables.

#### New state

```python
self._allocator: BlockAllocator
self._request_blocks: dict[str, list[int]] = {}
self._block_table_buffer: torch.Tensor  # (max_active_requests, num_blocks_per_seq), int32, on device
```

The `_block_table_buffer` is the single source of truth for the block table of every active request. It replaces the precomputed static block tables in `LiteEngine._fast_block_tables`.

#### New methods

```python
def ensure_blocks_for_requests(
    self,
    request_ids: list[str],
    token_counts: list[int],
) -> None:
    """Allocate additional blocks so each request can hold at least token_counts[i] tokens."""

def ensure_blocks(self, request_id: str, num_tokens: int) -> int:
    """Allocate additional blocks if request currently has fewer than ceil(num_tokens/block_size).
    Returns the number of newly allocated blocks."""

def free_request_blocks(self, request_id: str) -> None:
    """Free all blocks held by a request and clear its block-table row."""

def block_table_for_slot(self, slot_idx: int) -> torch.Tensor:
    """Return the device's block-table row for the request currently occupying slot_idx."""

def update_block_table_row(self, slot_idx: int, request_id: str) -> None:
    """Copy request block IDs into the shared block-table buffer row for slot_idx."""
```

`capture_prefix_entry` and `materialize_prefix_entry` are updated to operate on request block IDs instead of slot-based contiguous ranges:

- **Capture:** gather the used blocks via advanced indexing, e.g. `k_view[request_block_ids[:used_blocks]]`.
- **Materialize:** allocate blocks for the target request, then write the stored blocks back with `k_view[dst_block_ids[:used_blocks]] = entry.k_blocks[layer]`.

The `PrefixCacheEntry` dataclass can keep the same fields; the tensors stored inside it will simply be gathered in block-ID order rather than contiguous slices.

### Component 4: `compute_slot_mapping` Triton kernel

**Location:** `vllm/kernels/triton/compute_slot_mapping.py`

Responsibility: convert `(request_id, position)` into a linear slot index `block_id * block_size + offset` using the request's dynamic block table.

#### Kernel contract

Inputs:

- `query_start_loc`: `(num_reqs + 1,)` int32, token offsets per request.
- `positions`: `(num_tokens,)` int64, token positions.
- `block_table`: `(max_reqs, max_blocks_per_req)` int32.
- `block_size`: int.

Output:

- `slot_mapping`: `(num_tokens,)` int64; padding slots receive `PAD_SLOT_ID = -1`.

The kernel is needed for prefill where many tokens are processed at once. For decode (one token per request) the same kernel can be invoked with `num_tokens == batch_size`, or the mapping can be computed cheaply on the CPU. To keep the path uniform, both prefill and decode will use the kernel.

### Component 5: `InputBatchBuilder` updates

**Location:** `vllm/engine/input_batch_builder.py`

Changes:

1. Remove all uses of `slot_idx * max_model_len + position` for `slot_mapping`.
2. Build `slot_mapping` by calling the new Triton kernel with `positions` and the request block tables.
3. Continue using `KVBlockManager.block_table_for_slot(slot_idx)` to fetch the device block-table row; the row content is now dynamic and reflects the request currently occupying the slot.
4. Keep `slot_idx` for indexing the shared block-table buffer and the fast-path scratch tensors (`_fast_input_ids`, `_fast_positions`, etc.).

For the decode fast path, `build_decode_fast` must update `fast_slot_mapping` values using the request's dynamic block table before invoking the executor.

### Component 6: `LiteEngine` and runtime assembly

**Location:** `vllm/engine/lite_engine.py`, `vllm/engine/initialization/runtime_component_factory.py`, `vllm/engine/runtime_factory.py`

Changes:

1. `LiteEngine` constructs `FlatKVCacheAllocator` and `BlockAllocator` instead of `KVCacheAllocator`.
2. The static precomputed `_fast_block_tables` tensor in `LiteEngine` is removed; the block-table buffer lives inside `KVBlockManager`.
3. `LiteRuntimeAssembler.assemble` and `RuntimeAssemblyContext` accept a `block_allocator` argument and pass it to `LiteRuntimeFactory.build`.
4. `LiteRuntimeFactory.build` constructs `KVBlockManager` with the allocator.

### Component 7: Backend lifecycle integration

**Location:** `vllm/engine/backend/lite_single_gpu.py`, `vllm/engine/runtime_controller.py`

Changes:

1. **Block allocation timing:** before a step executes, `RuntimeController` calls `kv_block_manager.ensure_blocks_for_requests(request_ids, token_counts)` for all scheduled requests. Token counts are derived from the `StepPlan`:
   - Prefill: `start_pos + chunk_len` for each selected prefill request.
   - Decode: `seq_len + 1` for each selected decode request.
2. **Block freeing:** when a request finishes or is aborted, `LiteSingleGpuBackend` calls `kv_block_manager.free_request_blocks(request_id)` in addition to `scheduler.free_request`.
3. **Prefix cache:** `_store_prefix_cache_entry` and `_materialize_prefix_cache_entry` pass `request_id` to `KVBlockManager` instead of `slot_idx`.

### Component 8: Block zeroing

Freshly allocated blocks may contain data from previously completed sequences. Before a block is used by a new request, it must be zeroed.

Implementation: after `BlockAllocator.allocate` returns new IDs, zero them across all layers with advanced indexing:

```python
for k_view, v_view in kv_caches:
    k_view[new_block_ids] = 0
    v_view[new_block_ids] = 0
for ks_view, vs_view in kv_scale_caches:
    if ks_view is not None:
        ks_view[new_block_ids] = 0
    if vs_view is not None:
        vs_view[new_block_ids] = 0
```

Because `new_block_ids` is typically small (decode: ≤1 per request; prefill chunk: ≤32), this is cheaper than a per-block Python loop.

## Memory Sizing

`RuntimePlanner` continues to compute:

```text
num_blocks_per_seq = max_model_len // block_size
num_total_blocks   = max_active_requests * num_blocks_per_seq
```

The flat buffer size uses the same total number of blocks. The only new overhead is one reserved null block, which is negligible.

`compute_kv_theory_bytes` already supports per-layer specs; it will be used for the actual allocation size and for the startup memory audit.

## Data Flow Example

### Admission

1. `RequestScheduler.admit_queued_requests` assigns a `slot_idx` as today.
2. No blocks are allocated yet.

### First prefill step

1. `StepScheduler` selects the request for prefill with `chunk_len`.
2. `RuntimeController` calls `kv_block_manager.ensure_blocks_for_requests([request_id], [start_pos + chunk_len])`.
3. `KVBlockManager` allocates the needed block IDs, zeros them, and calls `update_block_table_row(slot_idx, request_id)`.
4. `InputBatchBuilder.build_prefill` builds `positions`, calls `compute_slot_mapping`, and passes the dynamic `block_tables` to the executor.
5. `reshape_and_cache` writes K/V using the dynamic `slot_mapping`.

### Decode step

1. `StepScheduler` selects the request for decode.
2. Backend ensures one more block if the decode token crosses a block boundary.
3. `KVBlockManager.update_block_table_row(slot_idx, request_id)` refreshes the shared block-table buffer.
4. `InputBatchBuilder.build_decode_batch` computes `slot_mapping` from the request's block table.
4. `PagedAttention` reads the same block IDs from `block_tables`.

### Finish

1. Backend calls `kv_block_manager.free_request_blocks(request_id)`.
2. `RequestScheduler.free_request` releases the `slot_idx`.

## Testing Plan

### Unit tests

- `FlatKVCacheAllocator` returns views with correct shape and shared underlying storage.
- Per-layer views are contiguous and strides match the old per-layer allocation.
- `BlockAllocator` allocate/free/free count semantics; null block is never allocated.
- `KVBlockManager` ensures blocks grow monotonically and frees them correctly.
- Prefix cache capture/materialize produces numerically identical results before and after the change (using a small model).

### Kernel tests

- `compute_slot_mapping` matches a CPU reference for:
  - single-token decode,
  - short prefill,
  - chunked prefill crossing block boundaries,
  - padding slots.

### Integration / regression

- `bash tests/run_regression_suite.sh`
- `SKIP_A_TIER=1 bash tests/run_inference_correctness_regression.sh`
- `uv run python tests/e2e_full_benchmark.py`
- Manual spot checks on:
  - 0-token prompt edge cases,
  - request abort mid-prefill,
  - prefix cache hit/partial hit/miss.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Slot mapping kernel bug corrupts KV writes | CPU reference test + correctness regression |
| Dynamic block allocation exhausts pool mid-step | `ensure_blocks` raises; backend skips the request instead of OOM |
| Block table buffer not updated before fast path | Add an invariant assertion in `build_decode_fast`; remove static `_fast_block_tables` |
| Prefix cache materialization with non-contiguous blocks slower | Use advanced indexing; measure in e2e benchmark |
| Per-layer view strides differ from old allocation and break kernel assumptions | Unit test stride equality for uniform models; explicit dtype/stride assertions in allocator tests |

## Rollout

1. Implement `FlatKVCacheAllocator`; compare memory audit and strides with old allocator in a throwaway branch.
2. Implement `BlockAllocator` and integrate into `KVBlockManager` while keeping static slot mapping as a fallback.
3. Add `compute_slot_mapping` kernel and switch `InputBatchBuilder`.
4. Remove static `_fast_block_tables` and update fast path.
5. Update prefix cache interface.
6. Run full regression and correctness suites.
7. Remove the old `KVCacheAllocator` code path once the new path is stable.

## Out of Scope for This Design

- Upstream v1 `BlockPool` / `KVCacheManager` / prefix caching.
- Cross-request block sharing.
- DeepSeek V4 Flash direct runtime KV management.
