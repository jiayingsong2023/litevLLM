# DeepSeek V4 Flash 异步 Expert Weight Prefetch 设计

## 1. 目标

把 DeepSeek V4 Flash GGUF 的 **cold-cache decode 吞吐**从当前 ~0.46 tok/s 提升到接近 **warm-cache ~1.6 tok/s**，方法是在单 token greedy decode 路径上，把下一层专家权重的 H2D copy 放到独立 CUDA stream，与当前层 compute 重叠。

## 2. 背景与根因

- cold-cache 慢不是因为 staging budget 不够。在 96 GB 统一内存上，runtime cache cap 已经达到 ~84 GiB（`vllm/model_executor/models/deepseek_v4_flash/model.py:1838-1865`）。
- warm-cache 快是因为 `FULL_RESIDENT=1` 把所有专家权重常驻 GPU。但默认不开启 full-resident，因为小显存 GPU 会 OOM。
- 当前代码在单 token decode 路径已经能算出下一层需要哪些专家（`_schedule_next_layer_expert_prefetch()` at `model.py:371`），但 prefetch 在 default stream 上执行，与 compute 串行。
- `gpu_weight_staging.py:759-770` 的 `prefetch_grouped_experts()` 已经支持 `stream` 参数，说明接口层面已经预留了异步能力。

## 3. 设计

### 3.1 Stager 持有 background stream

在 `DeepSeekV4FlashGPUWeightStager.__init__` 里创建：

```python
self._prefetch_stream = torch.cuda.Stream(device=self.device)
```

新增两个方法：

```python
def prefetch_grouped_experts_async(
    self,
    tensors: list[torch.Tensor],
    request: DeepSeekV4FlashExpertPrefetchRequest,
) -> torch.cuda.Event:
    """在 background stream 上 prefetch，返回 event 供 compute stream 同步。"""
    with torch.cuda.stream(self._prefetch_stream):
        self.prefetch_grouped_experts(tensors, request, stream=self._prefetch_stream)
    event = torch.cuda.Event()
    event.record(self._prefetch_stream)
    return event

def wait_for_prefetch(self, event: torch.cuda.Event | None) -> None:
    """让当前 compute stream 等待 prefetch 完成。"""
    if event is not None:
        torch.cuda.current_stream().wait_event(event)
```

### 3.2 单 token decode 路径触发下一层 prefetch

改动集中在 `vllm/model_executor/models/deepseek_v4_flash/model.py` 的单 token greedy decode 路径：

1. `_schedule_next_layer_expert_prefetch()`（`model.py:371`）返回的 request 改由 `stager.prefetch_grouped_experts_async()` 执行。
2. 返回的 `torch.cuda.Event` 存到当前的 decode step state（如 `self._pending_prefetch_event`）。
3. 在进入下一层 `forward` 前，调用 `stager.wait_for_prefetch(self._pending_prefetch_event)`。

这样每层 compute 都会自动 overlap 上一层的 H2D copy。

### 3.3 范围限制

- **只改单 token greedy decode 路径**（`model.py:687` 附近）。该路径有明确的 `token_id`，能准确预测下一层专家。
- **不改** `token_id=None` 的路径（`model.py:791`、`model.py:1122`）和 graph replay 路径，避免引入额外同步风险和 tensor identity 问题。
- 保留 `FASTINFERENCE_DEEPSEEK_V4_FLASH_ASYNC_PREFETCH` env 开关，默认 `1`，可设为 `0` 回退到当前行为。
- 该 env 名需要在 `vllm/engine/env_registry.py` 里注册为 `_tool_only`，以满足项目治理测试。

## 4. 数据流

```
Layer N compute on default stream
        │
        ├─► _schedule_next_layer_expert_prefetch() ──► prefetch on _prefetch_stream
        │                                                   │
        │                                                   ▼
        │                                           record Event E
        │
        ▼
Layer N+1 compute on default stream
        │
        ├─► wait_event(E)  (确保数据已就绪)
        │
        ▼
   consume staged payload
```

## 5. 正确性保障

- background stream 只做 H2D copy / `stage_*`，不做 compute。
- 每次消费 staged tensor 前，compute stream 必须 `wait_event(event)`。
- 如果 prefetch 还没完成，compute stream 自然阻塞，语义等价于当前串行行为。
- 单元测试覆盖：prefetch 真在 background stream、event 同步有效、prefetch 过的 payload 在 demand 路径命中。

## 6. 验证计划

1. **单元测试**：扩展 `tests/deepseek_v4_flash/test_expert_prefetch.py`
   - `test_prefetch_async_uses_background_stream`
   - `test_wait_for_prefetch_synchronizes_compute_stream`
   - `test_prefetched_payload_reused_on_demand`
2. **快速回归**：`bash tests/run_regression_suite.sh`
3. **真实性能**：`bash tests/run_deepseek_v4_flash_real_smoke.sh`，重点看 cold gate 的 `decode_tps_steady_state` 和 `usable_inference_metrics` 里的 cache hit/miss。
4. **正确性回归**：`SKIP_A_TIER=1 bash tests/run_inference_correctness_regression.sh`

## 7. 风险与回退

| 风险 | 缓解 |
|---|---|
| stream 顺序错误导致消费未就绪 tensor | 强制 `wait_event`；单元测试验证 |
| background copy 比 compute 快，收益有限 | 先跑真实 A/B，再决定是否叠加其他优化 |
| 小显存 GPU 行为变化 | 不改 full-resident 默认；提供 env 开关回退 |
| graph / batched 路径未覆盖 | 明确限制范围；后续可单独评估 |

## 8. 成功标准

- cold-cache gate (`--warmup-tokens 1`) 的 `decode_tps_steady_state` 从 ~0.46 提升到 **≥1.0 tok/s**（作为第一阶段目标，接近 warm 1.6）。
- warm-cache gate 不下降。
- 所有回归测试通过。
