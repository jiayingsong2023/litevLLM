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

新增两个方法（签名与现有 `prefetch_grouped_experts` 保持一致）：

```python
def prefetch_grouped_experts_async(
    self,
    tensors: DeepSeekV4FlashGroupedExpertTensors,
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

### 3.2 单 token decode 路径：提前一层触发 prefetch

核心改动在 `vllm/model_executor/models/deepseek_v4_flash/model.py` 的单 token greedy decode layer loop：

1. **进入 layer N compute 之前**，先等待并消费 `layer N` 的 prefetch event（即上一次迭代发起的 `layer N` prefetch）。
2. **layer N forward 返回后立即 enqueue `layer N+1` prefetch**：在单 token greedy decode 的外层 layer loop 中，调用 `layer.forward()` 后马上调用 `_schedule_next_layer_expert_prefetch()` 计算 `layer N+1` 的专家需求，并调用 `stager.prefetch_grouped_experts_async()` 在 background stream 上发起 prefetch。overlap 依赖 CUDA kernel 的异步执行以及 default stream 队列仍在运行；实现时必须确认这段路径没有 `torch.cuda.synchronize()` 或隐式同步。
3. 返回的 `torch.cuda.Event` 不存为 model 级状态，而是作为 **本次 layer loop 的局部变量**传给下一次迭代。

这样 `layer N+1` 的 H2D copy 与 `layer N` 在 default stream 上尚未完成的 compute 重叠，而不是在 `layer N` 结束后立即等待。

### 3.3 范围与开关

- **只改单 token greedy decode 路径**（`model.py:687` 附近）。该路径有明确的 `token_id`，能准确预测下一层专家。
- **不改** `token_id=None` 的路径（`model.py:791`、`model.py:1122`）和 graph replay 路径，避免引入额外同步风险和 tensor identity 问题。
- 保留 `FASTINFERENCE_DEEPSEEK_V4_FLASH_ASYNC_PREFETCH` env 开关，**默认 `0`**，仅在 smoke/A-B 命令里显式打开；跑出稳定收益后再评估默认启用。
- 该 env 名需要在 `vllm/engine/env_registry.py` 里注册为 `_tool_only`，以满足项目治理测试。

## 4. 数据流

```
Loop iteration for layer N:

  ┌─► wait_event(E_N)          # 确保 layer N 的 prefetch 已就绪
  │
  ▼
Layer N compute on default stream
  │
  ├─► _schedule_next_layer_expert_prefetch(N+1)
  │       │
  │       ▼
  │   prefetch on _prefetch_stream
  │       │
  │       ▼
  │   record Event E_{N+1}
  │
  ▼
pass E_{N+1} to next iteration
```

第一层的 prefetch event 为空（不等待），最后一层的 prefetch 无后续消费（可忽略）。

## 5. 正确性保障

- background stream 只做 H2D copy / `stage_*`，不做 compute。
- 每次消费 staged tensor 前，compute stream 必须 `wait_event(event)`。
- 如果 prefetch 还没完成，compute stream 自然阻塞，语义等价于当前串行行为。
- `event` 作为局部变量在单次 layer loop 内传递，不进入 model 状态，避免多请求共享 model 时串扰。
- 单元测试覆盖：prefetch 真在 background stream、event 同步有效、prefetch 过的 payload 在 demand 路径命中。

## 6. 验证计划

1. **单元测试**：扩展 `tests/deepseek_v4_flash/test_expert_prefetch.py`
   - `test_prefetch_async_uses_background_stream`
   - `test_wait_for_prefetch_synchronizes_compute_stream`
   - `test_prefetched_payload_reused_on_demand`
2. **A/B 性能验证**：在 `tests/run_deepseek_v4_flash_real_smoke.sh` 的 cold gate 上跑两轮，保存 profile JSON：
   - `FASTINFERENCE_DEEPSEEK_V4_FLASH_ASYNC_PREFETCH=0`
   - `FASTINFERENCE_DEEPSEEK_V4_FLASH_ASYNC_PREFETCH=1`
   - 对比 `decode_tps_steady_state`、staging cache hit/miss、`streamed_bytes`、`prefetch_*`  counters、`layer_moe` phase time。
3. **快速回归**：`bash tests/run_regression_suite.sh`
4. **正确性回归**：`SKIP_A_TIER=1 bash tests/run_inference_correctness_regression.sh`

## 7. 风险与回退

| 风险 | 缓解 |
|---|---|
| stream 顺序错误导致消费未就绪 tensor | 强制 `wait_event`；单元测试验证 |
| background copy 比 compute 快，收益有限 | 先跑 A/B 证明收益，默认保持关闭 |
| 多请求共享 model 时 event 串扰 | event 作为局部变量，不存入 model state |
| graph / batched 路径未覆盖 | 明确限制范围；后续可单独评估 |

## 8. 成功标准

- 在 `FASTINFERENCE_DEEPSEEK_V4_FLASH_ASYNC_PREFETCH=1` 的 cold-cache A/B 中，`decode_tps_steady_state` 相对 `=0` 有 **可测量的提升**（目标 ≥0.2 tok/s 或相对提升 ≥30%）。
- A/B 必须保存 profile JSON，并对比 `layer_moe` phase time、staging `streamed_bytes`、prefetch hit/miss counters；如收益不达预期，这些指标用于区分是 H2D 未 overlap，还是 CPU payload materialization 成了瓶颈。
- warm-cache gate 不下降。
- 所有回归测试通过。
- 默认行为不变（`ASYNC_PREFETCH=0`），收益验证通过后再决策是否默认开启。
