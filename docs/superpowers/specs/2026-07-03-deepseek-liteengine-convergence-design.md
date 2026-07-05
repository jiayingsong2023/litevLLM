# DeepSeek V4 Flash 接入 LiteEngine 控制面设计

**Date:** 2026-07-03
**Author:** Kimi Code CLI
**Status:** Final implementation blueprint — v6

## 1. 背景与目标

### 1.1 当前问题

DeepSeek V4 Flash 是当前项目里唯一使用 `ModelAdapter.build_direct_runtime()` 的模型。`LiteEngine.__init__` 在检测到 `direct_runtime` 后直接提前返回：

```python
self.direct_runtime = self._build_direct_runtime()
if self.direct_runtime is not None:
    self.max_model_len = ...
    self.runtime_controller = None
    self.direct_runtime.prepare()
    return
```

这导致 DeepSeek 完全绕开了 `RuntimePlanner`、`RequestScheduler`、`StepScheduler`、`RuntimeController`、`OutputPipeline`、`RuntimeObserver` 等整个控制面。`AsyncLLM.generate()` 也直接调用 `direct_runtime.generate()`。

### 1.2 设计目标

**统一控制面，保留模型本地数据面。**

- `LiteEngine` / `RuntimeController` / `RequestScheduler` / `StepScheduler` / `OutputPipeline` / `RuntimeObserver` 必须统一。
- DeepSeek 的 compressed KV、raw sliding window、indexer、MoE weight staging、专用 kernels 仍留在 DeepSeek model/backend 内。
- 不改造标准 `KVBlockManager`；用薄 adapter 把 DeepSeek KV 生命周期桥接给 engine。
- `AsyncLLM` 不再直接调用 `direct_runtime.generate()`，所有请求走统一 engine queue。
- DeepSeek prefill/decode 遵循标准 request lifecycle：admit → ensure KV/state → prefill → decode steps → finish/abort/error → free。

### 1.3 非目标

- 不把 DeepSeek KV 塞进标准 `KVBlockManager`。
- 不强求 DeepSeek decode kernel 立刻输出 logits 以复用完整 `SamplingDriver`。
- 不引入通用 `WeightStager` 抽象；只暴露 DeepSeek 本地 budget/stats surface。
- 第一阶段不做 batched multi-token decode；只做单 token decode step。
- 不抽象通用 backend 协议；等第二个模型需要 token executor 再抽象。

---

## 2. 关键修正点（相对 v5）

| 问题 | v5 方案 | v6 修正 |
|---|---|---|
| custom branch  multimodal_processor = None | `LiteEngine.add_request()` 会无条件调用 `self.multimodal_processor.prepare_request()`，None 会炸 | custom branch 使用 `NullMultiModalProcessor`；`CustomRuntimeComponents` 加入 `multimodal_processor` 字段 |
| Phase 4 add_request 路径未明确 | 未提及 custom runtime 下 add_request 的构造流程 | 明确 add_request 仍能构造 request、执行 sampling validation、调用 no-op multimodal processor |
| Phase 4 验证缺 multimodal smoke | 未单独验证 | 加一项无图像请求的 `AsyncLLM.generate()` smoke |
| multimodal/LoRA 拒绝时机 | 放在 Phase 5 讨论 | Phase 4 统一入口后，拒绝应在 `add_request()` 早期发生；Phase 5 细化 |

---

## 3. 高层架构

```text
LLM / AsyncLLM
  └─ LiteEngine
       ├─ ModelAdapter (deepseek_v4_flash)
       │    ├─ detect() -> ModelCapabilities
       │    ├─ build_direct_runtime() -> None   # 返回 None，旧路径关闭
       │    └─ build_executors(...) -> CustomRuntimeComponents
       │
       ├─ RuntimePlanner / MemoryAuditor (standard)
       ├─ RequestScheduler / StepScheduler (standard)
       ├─ RuntimeController + LiteSingleGpuBackend (standard)
       │    ├─ prefill_executor: DeepSeekPrefillExecutor
       │    ├─ decode_executor: DeepSeekDecodeExecutor
       │    └─ kv_block_manager: DeepSeekKVLifecycleAdapter
       ├─ OutputPipeline (standard)
       └─ RuntimeObserver (standard)

DeepSeekV4FlashForCausalLM
  ├─ DeepSeekPagedKVCache           (model-local)
  ├─ DeepSeekV4FlashGPUWeightStager (model-local)
  ├─ DeepSeekV4FlashGPUBackend      (model-local)
  ├─ gpu_layers / kernels           (model-local)
  └─ request_id -> DeepSeekV4FlashGPURequestState map
```

---

## 4. CustomRuntimeComponents

新增 `vllm/engine/custom_runtime_components.py`：

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class CustomRuntimeComponents:
    """Adapter 提供的自定义 runtime 组件，用于替换标准 executor + KV manager + multimodal processor。"""
    prefill_executor: Any
    decode_executor: Any
    kv_block_manager: Any
    multimodal_processor: Any | None = None
```

当 `multimodal_processor` 为 `None` 时，factory 会自动填充 `NullMultiModalProcessor`。

### 4.1 NullMultiModalProcessor

新增 `vllm/engine/multimodal_processor.py` 中的 null 实现，或单独文件：

```python
class NullMultiModalProcessor:
    """No-op multimodal processor for models that do not support multimodal inputs."""

    def prepare_request(self, request_state: Any) -> None:
        pass

    def build_prefill_inputs(self, req_dicts: list[dict[str, Any]]) -> list[Any] | None:
        return None

    def get_multimodal_embeddings(self, mm_inputs: Any) -> None:
        return None

    def stats(self) -> dict[str, Any]:
        return {}

    def reset_stats(self) -> None:
        pass
```

这样 `LiteEngine.add_request()` 里 unconditional 的 `self.multimodal_processor.prepare_request(request_state)` 不会炸。

### 4.2 ModelAdapter.build_executors()

```python
class ModelAdapter(ABC):
    def build_executors(
        self,
        *,
        model: Any,
        model_config: Any,
        runtime_config: Any,
        observer: Any | None,
        **kwargs: Any,
    ) -> CustomRuntimeComponents | None:
        return None
```

DeepSeek adapter覆盖：

```python
class DeepSeekV4FlashAdapter(ModelAdapter):
    def build_executors(self, *, model, runtime_config, observer, **kwargs):
        kv_adapter = DeepSeekKVLifecycleAdapter(
            model=model,
            context_length=runtime_config.context_length,
            device=runtime_config.device,
            max_active_requests=runtime_config.max_active_requests,
        )
        return CustomRuntimeComponents(
            prefill_executor=DeepSeekPrefillExecutor(model=model, observer=observer),
            decode_executor=DeepSeekDecodeExecutor(model=model, observer=observer),
            kv_block_manager=kv_adapter,
            multimodal_processor=None,  # factory 会填 NullMultiModalProcessor
        )

    def build_direct_runtime(self, **kwargs) -> None:
        return None  # 明确关闭旧路径
```

### 4.3 LiteRuntimeFactory 分支

```python
class LiteRuntimeFactory:
    @classmethod
    def build(cls, context: RuntimeAssemblyContext) -> dict[str, Any]:
        custom = context.adapter.build_executors(
            model=context.model,
            runtime_config=context.runtime_config,
            observer=context.observer,
        )

        if custom is not None:
            prefill_executor = custom.prefill_executor
            decode_executor = custom.decode_executor
            kv_block_manager = custom.kv_block_manager
            input_batch_builder = None
            multimodal_processor = custom.multimodal_processor or NullMultiModalProcessor()
        else:
            kv_block_manager = KVBlockManager(...)
            input_batch_builder = InputBatchBuilder(...)
            multimodal_processor = LiteMultiModalProcessor(...)
            prefill_executor = PrefillExecutor(...)
            decode_executor = DecodeExecutor(...)

        ...
```

---

## 5. DeepSeek KV Lifecycle Adapter

### 5.1 接口定义

```python
class DeepSeekKVLifecycleAdapter:
    def __init__(
        self,
        *,
        model: DeepSeekV4FlashForCausalLM,
        context_length: int,
        device: torch.device,
        max_active_requests: int,
    ) -> None:
        self.model = model
        self.context_length = context_length
        self.device = device
        self.max_active_requests = max_active_requests

    @property
    def block_size(self) -> int:
        return self.model.raw_block_size()

    @property
    def num_blocks_per_seq(self) -> int:
        return self.model.num_raw_blocks_per_seq()

    @property
    def num_layers(self) -> int:
        return self.model.num_layers()

    def _ensure_request_state(self, request_id: str) -> None:
        self.model.ensure_request_state(
            request_id=request_id,
            context_length=self.context_length,
            device=self.device,
            max_active_requests=self.max_active_requests,
        )

    def ensure_blocks_for_requests(
        self,
        request_ids: list[str],
        token_counts: list[int],
    ) -> None:
        """
        backend 唯一会调用的入口。
        内部顺序：先 ensure_request_state，再 ensure_request_capacity。
        """
        for rid, total_tokens in zip(request_ids, token_counts):
            self._ensure_request_state(rid)
            self.model.ensure_request_capacity(rid, max(0, total_tokens - 1))

    def free_request_blocks(self, request_id: str) -> None:
        self.model.free_request_state(request_id)

    def stats(self) -> dict[str, Any]:
        return self.model.kv_stats()
```

### 5.2 Model 暴露的方法

```python
class DeepSeekV4FlashForCausalLM:
    def raw_block_size(self) -> int: ...
    def num_raw_blocks_per_seq(self) -> int: ...
    def num_layers(self) -> int: ...
    def device(self) -> torch.device: ...

    def ensure_request_state(
        self,
        request_id: str,
        context_length: int,
        device: torch.device,
        max_active_requests: int,
    ) -> None: ...

    def ensure_request_capacity(
        self,
        request_id: str,
        token_idx: int,
    ) -> None: ...

    def free_request_state(self, request_id: str) -> None: ...

    def get_request_state(self, request_id: str) -> DeepSeekV4FlashGPURequestState: ...

    def kv_stats(self) -> dict[str, Any]: ...
```

---

## 6. Request / Session 生命周期映射

```text
external request_id
       │
       ▼
  admitted (RequestScheduler)
       │
       ▼
  ensure_blocks_for_requests(request_id, total_tokens)
       │   └─> _ensure_request_state() 创建 DeepSeek state
       │   └─> ensure_request_capacity() 确保 KV block
       ▼
  prefill step ──► DeepSeekPrefillExecutor
       │              get existing state[request_id]
       │              run prefill_greedy_kernel()
       │              returns first next_token_id
       ▼
  decode step ──► DeepSeekDecodeExecutor
       │              get existing state[request_id]
       │              run single-token decode_greedy_kernel()
       │              returns next_token_id
       │
       ├── finish (EOS / max_tokens) ──► free_request_state(request_id)
       ├── abort ──► free_request_state(request_id)
       └── error ──► free_request_state(request_id)
```

---

## 7. Executor 契约与 Token-ID Fast Path

### 7.1 通用 token result 类型

新增 `vllm/engine/executor_result.py`：

```python
from typing import NamedTuple
import torch

class TokenPrefillResult(NamedTuple):
    next_token_ids: torch.Tensor           # (batch_size,)
    prefilled_tokens: list[int]            # 每个请求实际 prefill 了多少 token
    is_last_chunk: list[bool]              # DeepSeek 恒为 True

class TokenDecodeResult(NamedTuple):
    next_token_ids: torch.Tensor           # (batch_size,)
```

### 7.2 Executor 接口

```python
class DeepSeekPrefillExecutor:
    def execute(
        self,
        request_ids: list[str],
        scheduler: Any,
        chunk_len: int,
    ) -> TokenPrefillResult: ...

class DeepSeekDecodeExecutor:
    def execute_batch(
        self,
        request_ids: list[str],
        scheduler: Any,
    ) -> TokenDecodeResult: ...
```

### 7.3 Backend 的 token-id fast path

```python
prefill_result = self.prefill_executor.execute(...)
if isinstance(prefill_result, TokenPrefillResult):
    for i, rid in enumerate(request_ids):
        req = self.scheduler.get_request(rid)
        req.seq_len += prefill_result.prefilled_tokens[i]
        token_id = int(prefill_result.next_token_ids[i].item())
        req.generated_ids.append(token_id)
        req.is_prefill = False
        self.scheduler.transition_to_decode(rid)
        self._process_completion(rid, token_id, results)
else:
    logits, _, is_last_chunk_flags = prefill_result
    ...
```

Decode 同理：

```python
decode_result = self.decode_executor.execute_batch(...)
if isinstance(decode_result, TokenDecodeResult):
    for i, rid in enumerate(request_ids):
        req = self.scheduler.get_request(rid)
        req.seq_len += 1
        token_id = int(decode_result.next_token_ids[i].item())
        req.generated_ids.append(token_id)
        self._process_completion(rid, token_id, results)
else:
    logits, _ = decode_result
    ...
```

### 7.4 单 token decode step

```python
def execute_batch(self, request_ids, scheduler):
    next_token_ids = []
    for rid in request_ids:
        state = self.model.get_request_state(rid)
        token = self.model.decode_single_token(state)
        next_token_ids.append(token)
    return TokenDecodeResult(
        next_token_ids=torch.tensor(next_token_ids, dtype=torch.long, device=self.model.device()),
    )
```

### 7.5 Observer 不重复

DeepSeek executor 只报特有事件：

```python
observer.on_deepseek_stager_hit(request_id=rid, expert_id=eid)
observer.on_deepseek_stager_miss(request_id=rid, expert_id=eid)
observer.on_deepseek_kv_blocks_allocated(family=family, count=count)
observer.on_deepseek_decode_single_token_latency_us(latency_us)
```

---

## 8. Abort / Background Error Cleanup

### 8.1 新增 backend 释放入口

```python
def free_request_resources(self, request_id: str) -> None:
    try:
        self.kv_block_manager.free_request_blocks(request_id)
    except Exception:
        logger.exception("Failed to free KV blocks for request %s", request_id)
```

### 8.2 LiteEngine 调用路径

```python
def abort_request(self, request_id: str) -> None:
    self.scheduler.abort_request(request_id)
    if self.runtime_controller is not None:
        self.runtime_controller.backend.free_request_resources(request_id)

def handle_background_error(self, exc: BaseException) -> None:
    request_ids = self.scheduler.request_ids()
    self.observer.on_background_error(exc, request_ids)
    for request_id in request_ids:
        self.scheduler.publish_exception(
            request_id,
            exc if isinstance(exc, BackgroundLoopError) else BackgroundLoopError(str(exc)),
        )
        if self.runtime_controller is not None:
            self.runtime_controller.backend.free_request_resources(request_id)
        self.scheduler.free_request(request_id)
```

---

## 9. 关闭 direct_runtime

```python
class DeepSeekV4FlashAdapter(ModelAdapter):
    def build_direct_runtime(self, **kwargs) -> None:
        return None
```

`LiteEngine.__init__`：

```python
self.direct_runtime = self._build_direct_runtime()
custom_components = self.adapter.build_executors(...)

if custom_components is not None:
    assert self.direct_runtime is None, (
        "Custom executors and direct_runtime cannot coexist"
    )
    self._use_custom_runtime = True
else:
    self._use_custom_runtime = False

# 统一继续初始化控制面
self.execution_policy = select_loadtime_policy(...)
...
self.runtime_controller = runtime_components["runtime_controller"]
```

`stats()` / `reset_stats()`：

```python
def stats(self) -> dict[str, Any]:
    return self.runtime_controller.stats()

def reset_stats(self, *, clear_prefix_cache: bool = False) -> None:
    self.runtime_controller.reset_stats(clear_prefix_cache=clear_prefix_cache)
```

---

## 10. AsyncLLM 入口与 add_request 路径

### 10.1 删除 direct_runtime 特判

```python
async def generate(self, prompt, sampling_params, request_id, ...):
    lora_id = ...
    self.engine.add_request(
        request_id,
        prompt,
        sampling_params,
        lora_id=lora_id,
        lora_request=lora_request,
        multi_modal_data=multi_modal_data,
    )
    self.driver.notify_new_work()
    async for output in self.engine.get_request_stream(request_id):
        yield output
```

### 10.2 add_request 在 custom runtime 下仍完整工作

`LiteEngine.add_request()` 流程：

1. 构造 `RequestState`。
2. 调用 `self.multimodal_processor.prepare_request(request_state)` → NullMultiModalProcessor no-op。
3. 执行 sampling validation（DeepSeek greedy 检查）。
4. 加入 scheduler queue。

### 10.3 Sampling 限制前置

```python
def _validate_deepseek_sampling(self, sampling_params: SamplingParams) -> None:
    if sampling_params.temperature != 0.0:
        raise ValueError("DeepSeek V4 Flash supports greedy sampling only")
    if sampling_params.top_p != 1.0:
        raise ValueError("...")
```

### 10.4 multimodal / LoRA 拒绝

Phase 4 统一入口后，`add_request()` 应早期拒绝：

```python
if multi_modal_data is not None:
    raise ValueError("DeepSeek V4 Flash does not support multimodal inputs")

if lora_request is not None:
    raise ValueError("DeepSeek V4 Flash does not support LoRA")
```

Phase 5 再考虑是否真正支持（大概率继续拒绝）。

---

## 11. Admission Cap 与 Budget 阶段划分

### 11.1 Phase 4：保守静态 admission cap

```python
def _derive_conservative_max_active_requests(
    self,
    per_request_kv_bytes: int,
    per_request_staging_bytes: int,
) -> int:
    total_gpu_bytes = get_available_gpu_memory()
    reserved_bytes = self.runtime_config.reserved_memory_bytes
    per_request_bytes = per_request_kv_bytes + per_request_staging_bytes
    cap = (total_gpu_bytes - reserved_bytes) // max(1, per_request_bytes)
    return min(cap, self.runtime_config.kv_max_active_requests)
```

### 11.2 Phase 5：RuntimePlanner 精确预算

- 把 `estimate_kv_bytes` / `estimate_staging_bytes` 接入 `RuntimePlanner`。
- 支持根据实际 GPU 预算动态调整 `max_active_requests` 和 context length。
- 接入 `MemoryAuditor` 运行时校验。

---

## 12. 迁移阶段（v6 最终版）

### 阶段 1：DeepSeekKVLifecycleAdapter + request/session map

- `DeepSeekV4FlashForCausalLM` 维护 `request_id -> DeepSeekV4FlashGPURequestState` map。
- 新增 model 方法：`ensure_request_state`、`ensure_request_capacity`、`free_request_state`、`get_request_state`、`kv_stats`、`raw_block_size`、`num_raw_blocks_per_seq`、`num_layers`、`device`。
- 新增 `CustomRuntimeComponents` dataclass，含 `multimodal_processor` 字段。
- 新增 `NullMultiModalProcessor`。
- 新增 `DeepSeekKVLifecycleAdapter`，构造时接收 `context_length` / `device` / `max_active_requests`。
- `ensure_blocks_for_requests()` 内部顺序：先 `_ensure_request_state()`，再 `ensure_request_capacity()`。
- 不改动 `LiteEngine` early return；不改动 `AsyncLLM`。
- 验证：单元测试通过，existing regression 不炸。

### 阶段 2：DeepSeek executor pair + 单 token decode wrapper

- 新增 `vllm/engine/executor_result.py`：`TokenPrefillResult`、`TokenDecodeResult`。
- 新增 `DeepSeekPrefillExecutor` / `DeepSeekDecodeExecutor`。
- `DeepSeekPrefillExecutor.execute()` 返回 `TokenPrefillResult`。
- `DeepSeekDecodeExecutor.execute_batch()` 返回 `TokenDecodeResult`（单 token）。
- 在 model/backend 里加 `decode_single_token(state)` wrapper。
- Executor 只报 DeepSeek 特有 observer 事件。
- 不接入 backend；先用单元测试验证 executor 输出正确。

### 阶段 3：LiteRuntimeFactory custom branch（仅加能力，不启用 DeepSeek）

- `ModelAdapter` 加 `build_executors()` 钩子，默认返回 `None`。
- `LiteRuntimeFactory` 检测 custom executors，分支创建 runtime components。
- `LiteRuntimeAssembler` 把 `model_adapter` 传入 factory。
- DeepSeek adapter 暂不覆盖 `build_executors()`，保持走 `direct_runtime`。
- 验证：标准模型无回归；factory custom branch 可通过测试 double 验证。

### 阶段 4：关闭 direct_runtime + 启用 DeepSeek custom executors + 统一 AsyncLLM + admission cap

- DeepSeek adapter `build_direct_runtime()` 返回 `None`。
- DeepSeek adapter 覆盖 `build_executors()`，返回 `CustomRuntimeComponents`（`multimodal_processor=None`，factory 填 `NullMultiModalProcessor`）。
- `LiteEngine.__init__`：
  - 移除 early return；
  - 检测到 custom executors 时 assert `direct_runtime is None`；
  - `stats()` / `reset_stats()` 不再检查 direct_runtime。
- `LiteEngine.add_request()` 在 custom runtime 下仍能完整构造 request、执行 sampling validation、调用 no-op multimodal processor。
- `LiteEngine.add_request()` 早期拒绝 multimodal / LoRA。
- `AsyncLLM.generate()` 删除 `direct_runtime` 分支。
- `LiteSingleGpuBackend` 加 token-id fast path。
- `LiteSingleGpuBackend` 加 `free_request_resources()`。
- `LiteEngine.abort_request()` 和 `handle_background_error()` 调用 `runtime_controller.backend.free_request_resources()`。
- 在 `RequestScheduler` 和 DeepSeek KV pool 创建前落地保守静态 admission cap。
- 验证：
  - DeepSeek correctness regression 和 e2e 通过；
  - 标准模型无回归；
  - 无图像请求的 `AsyncLLM.generate()` smoke 通过，防止 `None.prepare_request` 回归。

### 阶段 5：RuntimePlanner budget / observer / stats 对齐

- model 暴露完整 `estimate_runtime_bytes()` / `stats()`。
- `RuntimePlanner` 调用 DeepSeek budget surface。
- executor 补齐 DeepSeek 特有 observer 事件。
- 验证：OOM 风险降低，observer 能看 DeepSeek 状态。

### 阶段 6：batched decode 优化

- 让 `StepScheduler` 把多个 DeepSeek decode-ready 请求打包。
- `DeepSeekDecodeExecutor.execute_batch()` 同时处理多条请求，复用 `decode_single_token`。
- finished 的请求不再进入下一步 batch（不 padding）。
- 验证：多并发 DeepSeek 请求吞吐提升。

---

## 13. 测试计划

### 13.1 单元测试

- `DeepSeekKVLifecycleAdapter.ensure_blocks_for_requests()` 内部先创建 state 再 ensure capacity。
- Adapter 不调用 scheduler/model 反向方法。
- `NullMultiModalProcessor.prepare_request()` / `stats()` / `reset_stats()` no-op。
- `DeepSeekPrefillExecutor.execute()` 返回正确的 `TokenPrefillResult`。
- `DeepSeekDecodeExecutor.execute_batch()` 返回正确的 `TokenDecodeResult`。
- `LiteSingleGpuBackend` 对 token result 的处理正确更新 `req.seq_len` 和 `generated_ids`，传给 `_process_completion()` 的是 int。
- `LiteEngine.abort_request()` 调用 `runtime_controller.backend.free_request_resources()`。
- `LiteEngine.handle_background_error()` 对每个 affected request 调用 `free_request_resources()`。
- DeepSeek executor 不重复触发标准 observer 事件。
- DeepSeek adapter `build_direct_runtime()` 返回 `None`。
- `LiteEngine.stats()` / `reset_stats()` 不再走 direct_runtime。

### 13.2 集成测试

- `AsyncLLM.generate()` 对 DeepSeek 输出流式 `RequestOutput`（无 multimodal 数据）。
- `LLM.generate()` 对 DeepSeek 输出完整结果。
- 非 greedy sampling params 在 `add_request()` 阶段被拒绝。
- multimodal / LoRA 请求在 `add_request()` 阶段被拒绝。
- abort 请求能正确释放 DeepSeek KV 和 state。
- background error 能正确释放 DeepSeek KV 和 state。

### 13.3 回归测试

- `bash tests/run_inference_correctness_regression.sh` DeepSeek Tier-B PASS。
- `uv run python tests/e2e_full_benchmark.py` DeepSeek direct-GGUF 完成。
- 标准模型（Gemma4/Qwen/Llama）regression 无回归。

### 13.4 性能测试

- 单请求 DeepSeek decode tps 不下降。
- 阶段 6 完成后，多并发 DeepSeek 请求 decode tps 提升。

---

## 14. 风险与缓解

| 风险 | 缓解 |
|------|------|
| custom branch multimodal_processor = None 导致 `add_request()` 炸 | 使用 `NullMultiModalProcessor`，`CustomRuntimeComponents` 含该字段 |
| direct_runtime 和 custom executors 共存 | DeepSeek adapter 返回 `None`；`LiteEngine` assert 不共存 |
| DeepSeek prefill 不能 chunk | executor 返回 `is_last_chunk=True`；scheduler 把它当 atomic prefill |
| `LiteSingleGpuBackend` 硬依赖 `KVBlockManager` 属性 | `DeepSeekKVLifecycleAdapter` 提供所需属性 |
| Executor 返回类型需要 backend 分支 | 用通用 `TokenPrefillResult` / `TokenDecodeResult`，engine 不依赖 DeepSeek |
| State 在 `ensure_blocks_for_requests()` 前未创建 | adapter 内部先 `_ensure_request_state()` |
| Adapter 反向依赖 scheduler/model | adapter 只接收固定 runtime 上下文 |
| Request state 在 abort/background error 时泄漏 | 新增 `backend.free_request_resources()`，被 engine 调用 |
| 标准模型被 custom branch 误伤 | `build_executors()` 默认返回 `None`，标准路径完全不变 |
| decode kernel 无法单步拆分 | 先加 `decode_single_token()` wrapper；如果拆不出，阶段 4 可推迟 |
| Observer 双计 | executor 只报特有事件，标准事件由 controller/backend 负责 |
| Admission cap 太晚生效 | 在 `RequestScheduler` 和 KV pool 创建前计算并应用 |

---

## 15. 结论

- **v6 已可作为实施蓝图**：NullMultiModalProcessor 补上最后一个真实缺口，custom branch 不会再因为 `None.prepare_request()` 炸。
- **下一步可执行**：从阶段 1 开始，先写 `CustomRuntimeComponents`、`NullMultiModalProcessor`、`DeepSeekKVLifecycleAdapter` 和 model request/session map，不动 engine 入口。
