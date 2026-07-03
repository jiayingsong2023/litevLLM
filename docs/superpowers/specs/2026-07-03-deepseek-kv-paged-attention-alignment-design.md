# DeepSeek V4 Flash KV 管理对齐 PagedAttention 设计

**Date:** 2026-07-03  
**Status:** Implemented — DeepSeek-local model-level paged cache

## 1. 背景与目标

DeepSeek V4 Flash 的 KV 形态和标准 PagedAttention 不同：

- raw KV 是 128-token sliding window ring buffer。
- compressed KV 按层使用不同压缩比例，ratio=4 和 ratio=128 的有效 row 数不同。
- indexer rows 只存在于 ratio=4 层。

因此本设计不把 DeepSeek 强行接入通用 `KVBlockManager` 或标准 PagedAttention kernel。目标是让 DeepSeek 获得 paged physical storage、按需增长、请求生命周期释放和跨请求复用，同时把 DeepSeek 特例保留在模型本地实现里。

## 2. 已落地的形态

当前实现位于 `vllm/model_executor/models/deepseek_v4_flash/compressed_kv.py`：

```text
DeepSeekV4FlashForCausalLM
  └─ model-level DeepSeekV4PagedKVCache pool
       ├─ _families["raw"]        -> _LazyPagedRows
       ├─ _families["compressed"] -> _LazyPagedRows
       ├─ _families["indexer"]    -> _LazyPagedRows
       └─ _request_states[request_id] -> raw_token_indices / compressed counts

DeepSeekV4FlashGPURequestState
  └─ DeepSeekV4PagedKVRequestCache(request_id, shared_pool)
```

`DeepSeekV4FlashForCausalLM` 按 `(context_length, head_dim, dtype, device, max_requests)` 缓存共享 pool。每个 request state 只绑定一个 request view，不再拥有一份独立 dense cache。batched decode 会按当前 batch size 传入 `max_requests`，避免多个同时活跃 request 争用只有单请求容量的 block ID 空间。

## 3. 核心决策

| 决策 | 当前实现 | 理由 |
|------|----------|------|
| 是否复用 `KVBlockManager` | 否 | DeepSeek raw/compressed/indexer 语义不同，泛化会污染标准路径。 |
| 物理存储 | `DeepSeekV4PagedKVCache` 下三组 `_LazyPagedRows` | raw、compressed、indexer 的行宽和生命周期不同，分开最小。 |
| allocator | 每组 `_LazyPagedRows` 内部一个 `BlockAllocator`，容量按 `max_requests * layers * blocks_per_layer` 计算 | 复用现有安全检查；不保留旧 `DeepSeekV4KVPageAllocator`。 |
| 请求生命周期 | model-level pool + request view | 多并发共享 backing chunks 和 freed block ids。 |
| raw window 元数据 | 保留 `raw_token_indices` | wrap 后必须靠真实 token index 判断窗口内有效槽。 |
| compressed counts | 每 request、每 layer 保存 | ratio=4 和 ratio=128 有效 row 数不同，不能合成全局 family 状态。 |
| block 清零 | 每次分配给请求前 `zero_()` | 防止旧请求数据泄漏。 |
| CUDA graph / prefix cache | 非目标 | 当前 direct runtime 先保证正确性和内存模型。 |

## 4. API 边界

`DeepSeekV4PagedKVRequestCache` 保持原 dense cache 的调用形态：

```python
append_raw(layer_idx, token_idx, key, value)
read_raw_window(layer_idx, token_idx, window)
append_compressed(layer_idx, token_idx, row, indexer_row=None)
read_compressed(layer_idx, row_indices=None, count=None)
read_indexer_rows(layer_idx, count=None)
free_request_blocks()
```

`gpu_layers.py` 继续通过 `state.raw_kv_cache` / `state.compressed_kv_cache` 调用这些方法。`state.kv_cache` 是主名字，旧别名保留用于兼容现有调用点。

`DeepSeekV4CompressedKVCache` 仍保留为 dense reference/cache path，用于参考实现和交叉验证；GPU direct path 使用 `DeepSeekV4PagedKVCache`。

## 5. 内部机制

### 5.1 Lazy physical chunks

`_LazyPagedRows` 不在初始化时分配 full context tensor。它只在某个 request/layer 写入到新 block 时创建 backing chunk：

```text
logical row/token -> block position -> physical block id -> chunk id + block offset
```

chunk 以 `torch.empty` 创建，具体 block 在交给请求前清零。

### 5.2 Freed block reuse

通用 `BlockAllocator.free()` 保持 FIFO 语义，不为 DeepSeek 改全局行为。DeepSeek pool 在 `_LazyPagedRows` 内维护本地 `_reuse_ids`：请求结束后 block id 留在模型级 pool 的本地复用栈，下一请求优先拿这些 id，并在复用前清零。

这是 grow-only pool：已创建的 backing chunks 不释放给系统 allocator，避免并发请求之间反复分配大 tensor；请求释放的是 block ownership 和 per-request metadata。代价是某次长请求撑大的 pool 会保留到 model/runtime 清理。

### 5.3 Raw sliding window

raw family 写入 `token_idx % raw_window` 对应的逻辑槽位，同时更新 request-local `raw_token_indices[layer, slot]`。读取窗口时按真实 token index 过滤和排序，再 gather 对应槽位。

### 5.4 Compressed / indexer rows

compressed rows 由 `compressed_counts_cpu[layer_idx]` 决定追加位置。ratio=4 层可写 indexer rows；ratio=128 层没有 indexer cache，`read_indexer_rows()` 直接返回 shape `(0, indexer_head_dim)` 的空张量。

## 6. 内存预算边界

`DeepSeekV4FlashMemoryPolicy.estimate_context_bytes()` 仍表示单个活跃 request 的 context worst-case 估算。model-level pool 的物理 block ID 容量由 `DeepSeekV4PagedKVCache(max_requests=...)` 扩到并发请求数；如果上层要做严格并发显存预算，应把 per-request context estimate 乘以 active request 数，或引入单独的 runtime concurrency budget 字段。当前实现不改 runtime budget API。

## 7. 已清理的旧设计

旧 `DeepSeekV4KVPageAllocator`、`DeepSeekV4KVPagePool`、`DeepSeekV4PageRef` 已删除。它们只做逻辑 page 坐标换算，没有真实物理 block ownership，和当前 paged pool 概念重复。

`DeepSeekV4PagedKVCache.raw_keys/raw_values` 不再是存储字段。它们是兼容属性，返回空张量视图用于 dtype/device/shape 查询；真实 raw 数据在 `_families["raw"]` 的 backing chunks 中。

## 8. 测试覆盖

当前覆盖点：

- `BlockAllocator` 保留 FIFO free 顺序和 double-free/非法 id 检查。
- `DeepSeekV4CompressedKVLayout` 只保留 layout 能力，不再引用旧 page allocator。
- `DeepSeekV4PagedKVCache` lazy chunk 分配、raw window wrap、compressed/indexer read、free 后清零复用。
- request views 跨请求共享同一个 model-level pool，且 pool 可按 `max_requests` 支撑并发请求容量。
- ratio=128 层读取 indexer rows 返回空张量。
- runtime write helper 继续走 paged cache API。

## 9. 仍非目标

- 把 DeepSeek 接入通用 `KVBlockManager`。
- 改造标准 PagedAttention kernel 适配 DeepSeek compressed attention。
- DeepSeek prefix cache / COW。
- CUDA graph 专门处理。
- 抽象通用 FlatBufferAllocator。
