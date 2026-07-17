# FastInference Runtime Hardening 与服务正确性设计

- 状态：In progress
- 日期：2026-07-17

## 1. 背景

FastInference 已经形成稳定的 lite-only 架构：单 GPU、纯 Python + Triton，
`LiteEngine`/`RuntimeController` 负责通用控制面，模型特定执行与 KV 语义由
adapter 和 executor 持有。该方向不需要重写。

当前上线风险集中在错误路径和服务边界：请求终态没有统一所有权、后台 step
异常可能让 stream 永久等待、上下文长度可以超过物理 block-table 行、保留的
null block 未计入容量，以及 HTTP/多模态入口在成功响应和资源限制上不够严格。

本设计只收口这些已确认问题。性能改动必须先测量，kernel 文件不因行数而重写。

## 实施状态（2026-07-17）

- P0-A 已完成 CPU 覆盖：总上下文 admission 校验、block pool null block、
  `ensure_blocks()` fail-fast 与 paged-attention 读侧行边界 mask 已实现。
- P0-B 已完成 CPU 覆盖：统一 backend teardown、fatal 首因/worker 锁内清理、
  queue identity stream 清理和 duplicate ID 守卫已实现。消费者取消与 submit/stream
  分离仍属于 P1。
- P0-C 已完成服务默认边界：loopback 默认绑定、debug 默认关闭、解析前 request body
  上限、受限 data URI 图像、以及 pinned `trust_remote_code` opt-in 已实现。HTTP 重定向
  和离线本地文件 opt-in 不实现，保持默认拒绝；托管 CI 已执行 CPU smoke、env registry、
  no-C++、Triton import 与 lite-import closure。全量 mypy 暂不设为 required：当前基线有
  519 个错误，必须在独立类型债工作中清零，不能以 ignore 规则伪造门禁。
- P1 已完成：`AsyncLLM.submit()` 与 `stream()` 已分离，HTTP 在 SSE 200 前 submit，
  stream 取消或异常退出会调用幂等 abort。chat template、参数/usage 契约与异步锁迁移
  已完成完整 history template、支持参数转发、未支持参数 400、usage/finish reason、
  admission 429/503 映射与 submit/abort 的 `asyncio.to_thread`。lifespan、PrefixCache
  默认禁用、未迭代 stream 的 `close_stream()`、health/ready、未知 adapter 与 tuning key
  fail-closed 已完成。PrefixCache 的显式字节预算只有在重新启用缓存时才需要实现。
- P2 已完成：已删除无调用方的 FP8 MoE kernel、旧 `KVCacheAllocator`、
  `kv_cache_interface`、`with_cancellation`、旧 GGUF weight mapper 和上游 workflow
  residue；int4 注释已与代码一致，atomic reduce 已声明为 tolerance-only。benchmark
  JSON 已携带 fingerprint，交错 A/B 工具要求至少三轮。TinyLlama ROCm 基线未证明
  block-table 更新是端到端瓶颈，因此未进行增量写入改造。Gemma4-12B FP8 的 M=1
  reference 和 M=4 parity 均于 2026-07-17 通过；保持默认 scale=1.0。model path
  policy hints 已移出 safetensors/model-construction loader。完整 correctness 回归于同日
  通过；最新 isolated e2e 为 26B aggregate/decode `5.52/11.15 tok/s`、31B
  `2.42/4.56 tok/s`、DeepSeek `0.37/1.78 tok/s`。26B 独立复测三次的 aggregate
  中位数为 `5.52 tok/s`，未据此进行热路径改造。

## 2. 目标与非目标

### 2.1 目标

- 建立可测试的请求生命周期与唯一 teardown 路径。
- 保证任何请求都不能读写其 block-table 行之外的 KV。
- 区分可恢复的请求错误和不可证明可恢复的执行错误。
- 保证异步 API 不阻塞事件循环，且 admission 失败发生在 HTTP 成功响应之前。
- 将多模态输入收口为显式、有限、默认安全的信任边界。
- 在无 GPU 的托管 CI 上强制执行最小架构与 CPU 冒烟门禁。
- 保持“共享控制面、模型本地数据面”的现有边界。

### 2.2 非目标

- 不引入多 GPU、分布式调度、C++/CUDA 扩展或新的插件框架。
- 不把 DeepSeek 特有 KV、GGUF 或 kernel 语义塞进通用 `KVBlockManager`。
- 不以移除 `RLock` 作为吞吐优化；单 GPU step 本来就必须串行。
- 不重写已验证 Triton kernel，不按文件长度机械拆分 god file。
- 不在缺少 profile 与端到端 A/B 证据时修改 prefill、block-table 或 pacing 策略。
- 不承诺尚未实现的完整 OpenAI API 兼容性。

## 3. 必须成立的不变量

| 范围 | 不变量 |
| --- | --- |
| 上下文 | `prompt_tokens + max_tokens <= max_model_len` |
| KV 容量 | 可分配 block 数不少于 `slots * blocks_per_seq`，另有 block 0 作为 null block |
| KV 访问 | kernel 读写均不得访问请求 block-table 行之外的元素 |
| 请求 ID | 一个存活请求 ID 最多对应一个 request、slot、stream 和 LoRA 引用 |
| stream | 创建后恰好收到一个终态：finished output 或 exception |
| 通知顺序 | 终态发布先于 scheduler/KV/LoRA 状态释放 |
| stream 所有权 | stream 字典项只由消费生成器的 `finally` 删除 |
| teardown | normal finish、abort、timeout、error 共用同一个资源释放入口 |
| LoRA | 每次 `on_request_added` 最终恰好对应一次 `on_request_removed` |
| fatal | fatal 后不再 admission 或执行新 step，readiness 返回失败 |
| HTTP body | 在 JSON 解析前拒绝超出服务配置上限的实际请求体 |
| PrefixCache | 缓存 clone 的 GPU 字节数必须纳入容量预算，不能只按 entry 数限制 |

## 4. 总体设计

```text
HTTP request
  -> parse and validate
  -> submit/admit under engine serialization
  -> return stream handle only after admission succeeds
  -> worker executes step
       -> publish terminal output/exception
       -> release request resources exactly once
  -> consumer iterator finally removes stream entry

Unexpected post-admission execution error
  -> worker enters fatal state under engine lock
  -> publishes exception to every live stream
  -> releases every request exactly once
  -> stops driver and rejects new admission
  -> external supervisor restarts the process
```

控制面继续由 `LiteEngine`、`RuntimeController`、scheduler 和通用 observer 组成。
adapter/executor 继续拥有模型能力、模型本地 KV 生命周期和 kernel 选择。新增接口应
表达生命周期，不应引入第二套调度器或抽象层。

## 5. 上下文与 KV 安全

### 5.1 Admission 校验

`LiteRequestBuilder.build()` 在 tokenizer 和多模态展开完成后校验：

```text
len(input_ids) + effective_max_tokens <= max_model_len
```

不再只校验 prompt 本身。现有 `max_tokens_cap` 可先参与采样参数归一化，但归一化后
的总长度仍必须满足上式；超限以 `RequestRejectedError` 拒绝，不静默缩短用户请求。

decode 侧保留第二道防线：当 `seq_len >= max_model_len` 时正常结束，finish reason
为 `length`，不得继续进入 attention。

### 5.2 Block 池容量

block 0 是永不分配的 null block，因此运行时容量必须为：

```text
num_total_blocks = 1 + max_active_requests * num_blocks_per_seq
```

`BlockAllocator` 继续只分配 `1..N-1`。测试必须能让所有 slot 同时占满完整上下文，
并证明没有少一块或整批失败。

### 5.3 Kernel 防御

写侧已有安全语义：越界 slot 映射为 `PAD_ID`，cache 写 kernel 对负 slot 直接返回。
读侧 `paged_attention.py` 仍需独立防御：

- host 侧传入 block-table 每行容量；
- block-table 和 cached pointer 两条读取路径都使用
  `block_idx < max_num_blocks_per_seq` mask；
- 越界 lane 不形成全局内存地址，不依赖 admission 校验兜底。

`KVBlockManager.ensure_blocks()` 不再把超额需求静默 `min()` 到行容量。超额表示上层
不变量被破坏，应在任何 kernel 启动前抛出带 request ID、requested blocks 和 row
capacity 的错误。

### 5.4 4096 上限

`RuntimePlanner` 的 4096 上限当前是隐式钳制。保留这一产品限制时，应将其变成显式
规划结果和启动 warning；若用户配置要求超过 4096，则启动时报错。不得让配置看似
生效但实际静默缩短。

## 6. 请求、stream 与资源所有权

### 6.1 唯一 teardown

execution backend 暴露一个幂等的公共释放入口，例如：

```python
def release_request(self, request_id: str) -> RequestState | None: ...
```

该入口按固定顺序释放模型本地资源、通用 KV、scheduler/slot 和 LoRA 引用，并返回
被移除的 request。normal finish、显式 abort、queue timeout 和 background error
都调用它；`LiteEngine` 不再复制资源释放步骤。

队列中尚未获得 slot 的请求没有 KV，但仍可能持有 LoRA 引用，因此也必须经过同一
入口。`reject_expired_queued_requests()` 只发现过期项并返回 ID/reason，不先从
scheduler 删除；controller 先发布异常，再调用统一 teardown。幂等只用于处理
cancellation/error 竞态，不能掩盖重复 request ID。

### 6.2 Stream 生命周期

stream queue 在 scheduler 接受唯一 request ID 时创建。完成很快、消费者稍后 attach
是合法时序，因此 `free_request()` 不删除 `_request_streams`。

`get_request_stream()` 先取得 queue 的强引用，再在 `finally` 中按 request ID 和 queue
identity 删除字典项。这样旧消费者的 `finally` 不会误删后来复用同一 ID 的 queue。

终态顺序固定为：

1. 发布 finished output 或 exception；
2. 调用唯一 teardown；
3. 消费者观察终态并退出；
4. 消费生成器的 `finally` 删除 stream 字典项。

`AsyncLLM` 在消费者取消、生成器 `aclose()` 或客户端断连时调用幂等 abort。仅删除
queue 不足以清理 scheduler、KV 和 LoRA。

submit/stream 分离后，还必须覆盖“已 admission、但从未开始迭代”的窗口。stream handle
应有显式 close/abort 路径；不能假定未启动的 async generator 会运行其 `finally`。

### 6.3 生命周期测试矩阵

以下五条路径都必须断言 stream 终止，并最终满足：stream 不存在、request 不存在、
slot 已归还、KV 已释放、LoRA active count 为零。

| 路径 | 终态 |
| --- | --- |
| 正常完成 | finished output |
| step 异常 | exception |
| 显式 abort | abort output |
| queue timeout | `RequestRejectedError` |
| 消费者取消/断连 | abort 后生成器退出 |
| admission 后从未迭代 | handle close/abort 后清理 |

### 6.4 重复 request ID

在 `RequestScheduler.add_request()` 写入任何字典或队列前 fail fast，并同时检查
`_requests` 与 `_request_streams`。后者可能在 request 资源释放后继续存活到消费者
`finally`。`enqueue_request()` 已经委托该入口，无需增加第二套判重逻辑。测试覆盖
running、queued 和等待 stream detach 的 ID；失败不得覆盖旧 request、创建第二个
queue、重复入队或改变 slot。

## 7. 错误分级与 fatal 状态

### 7.1 可恢复请求错误

以下错误只终止单个请求，服务继续：

- schema、模型名、采样参数或多模态输入无效；
- prompt/context 超限；
- queue 满或 queue timeout；
- adapter 明确声明的不支持能力；
- kernel 启动前、已证明没有修改共享运行时状态的验证错误。

### 7.2 Fatal 执行错误

请求进入 step 后出现的未预期异常默认 fatal，包括 `ExecutionStepError`、allocator
不变量破坏、HIP/Triton/device error。原因是 kernel 可能已部分写 KV，GPU context
也可能处于 sticky error；通用 teardown 不能证明后续推理正确。

不采用“连续 N 次异常后 fatal”。计数器无法区分两个干净失败和一次已污染状态的
失败。未来只有在某类错误被证明发生于 kernel 启动前，才能把它降级为请求错误。

### 7.3 Fatal 流程

`AsyncDriver` 捕获 step 异常后在 worker 线程重新取得 engine lock，同步调用
`engine.handle_background_error()`，然后退出循环；不再把清理回调投递给事件循环。
scheduler 的 `publish_output()`/`publish_exception()` 已负责将 queue 写入 marshal
回事件循环。

fatal 处理按以下顺序执行：

1. 原子地将 engine 标记为 fatal，保存首个根因；
2. 停止继续 step 和 admission；
3. 向所有 live stream 发布统一的 `BackgroundLoopError`；
4. 通过唯一 teardown 释放所有 request；
5. driver 退出，不在循环中 sleep 后重试；
6. library 模式暴露 sticky fatal 状态；server 模式使 readiness 与 liveness 明确失败，
   并按部署契约退出非零或由 liveness probe 重启；
7. 不把“readiness 失败”误当作 systemd/容器一定会重启进程的机制。

后续错误不得覆盖首个根因。关闭流程可重复调用且不能再次发布终态。

## 8. 异步 admission 与 HTTP 语义

### 8.1 不阻塞事件循环

当前 `AsyncLLM.generate()` 在事件循环线程同步获取 `threading.RLock`，锁内还可能执行
tokenization、远程读取和 PIL decode。第一阶段将 add/abort/stats 等锁内同步调用放入
`asyncio.to_thread()`，避免事件循环等待 GPU step。

后续只有 profile 证明有必要时，才把 tokenization/多模态预处理移到 engine lock
之外。移动前必须定义 tokenizer、adapter 和 LoRA registry 的线程安全边界。

### 8.2 Admission 先于 StreamingResponse

异步生成器在第一次迭代前不会执行，因此不能把“提交请求”和“消费输出”藏在同一个
lazy generator 中。`AsyncLLM` 应拆为两个概念：

```text
submit(...) -> admitted stream handle
stream handle -> async iterator of RequestOutput
```

FastAPI endpoint 先 await `submit()`。只有 admission 成功后才构造
`StreamingResponse`，避免错误请求先返回 HTTP 200 再断开 SSE。

### 8.3 Chat API 最小兼容契约

- 请求使用 Pydantic/FastAPI schema，不直接信任任意 dict。
- `model` 必须与已加载模型或明确 alias 匹配。
- 完整 messages 历史通过 tokenizer chat template 生成 prompt。
- 支持的采样参数完整转发；未支持的 `top_p`、`top_k`、stop、penalty 等明确返回
  HTTP 400，不再静默忽略。
- queue/full-capacity 映射为 429，fatal/not-ready 映射为 503，输入错误映射为 400。
- streaming 和 non-streaming 都返回真实 finish reason 与 usage。
- 流式循环检查 `request.is_disconnected()`，在 `finally` 中 abort 未终态请求。
- FastAPI lifespan 在 shutdown 时调用 `AsyncLLM.shutdown()`。

在这些条件完成前，文档应称接口为“OpenAI-shaped”或列出兼容子集，不宣称完整兼容。

### 8.4 HTTP 服务暴露与调试端点

bundled server 默认绑定 `127.0.0.1`。对外绑定必须是显式操作，并由反向代理提供
认证和限流；本项目不新增自定义认证系统。

`/debug/*` 默认关闭，通过显式 server/TOML 开关启用；GET stats 与 POST reset 同受
该开关保护。后者会修改统计并可清 prefix cache，不能暴露在默认公网 listener。

HTTP body 在调用 `request.json()` 前按实际已接收字节限制，且不能只依赖
`Content-Length`。这条限制覆盖 text、JSON schema 和 data URI，防止图像解码限制前
就被大 body 占满内存。

## 9. 多模态输入信任边界

默认服务模式只允许受限 `data:` URI：

- 只接受允许列表内的 image MIME；
- base64 使用严格校验，并在解码前检查编码长度、解码后检查字节数；
- `Image.open()` 后、`.convert()` 触发完整解码前检查尺寸和总像素；
- 设置适合显存预算的 `Image.MAX_IMAGE_PIXELS`，将 decompression bomb warning
  视为请求错误；
- 任何限制都来自 TOML 的有类型配置，并有保守上限。

HTTP(S) 默认关闭。显式开启时必须设置连接/读取 timeout、`limit + 1` 响应体读取、
重定向次数上限，并对每次重定向后的目标重新执行协议、host 和 IP 策略校验。仅限制
初始 URL 不能阻止 SSRF。

`file://` 和裸本地路径在服务模式默认拒绝。若离线 `LLM` API 确有需求，可由独立
TOML flag 开启，并限制到显式配置的根目录；该 flag 不得被 HTTP server 自动继承。

模型加载同样是信任边界：`trust_remote_code` 默认关闭。需要远程模型实现时，由
TOML 显式启用并 pin revision/commit；配置构建和 loader 不得各自绕过该开关。

## 10. 配置、adapter 与通用层收口

这些工作排在 P0 正确性之后：

- `[tuning_keyvals]` 加载时按 registry 校验未知 key，拼错直接报错。
- KV dtype 在配置解析后使用枚举/字面量，不以字符串包含关系选择 kernel。
- adapter registry 对未知 architecture fail closed；不得静默回退 `LlamaAdapter`。
- 将 `gemma4_*` 运行配置和多模态模型分支移入 adapter-owned policy/processor。
- observer 只保留通用事件；模型专有计数通过 adapter observer hook 或结构化扩展注入。
- 同步 `AGENTS.md` 与 env registry：生产公共 env 入口只有
  `FASTINFERENCE_CONFIG`，废弃变量不能继续写成推荐控制项。

这里遵循一个约束：模型特例移出 engine 是移动所有权，不是删除行为。只有出现第二个
真实消费者时才抽取新的共享接口。

`VllmConfig` 已声明 runtime 配置和能力字段；只补显式 `runtime_observer` 字段并删除
不必要的 `object.__setattr__`，不为此重构配置体系。

## 11. CI 与性能治理

### 11.1 每个 PR 的托管 CPU 门禁

新增一个必过 job，至少执行：

```bash
uv run pytest tests/smoke tests/test_project_governance.py \
  tests/test_env_registry.py
uv run python tools/pre_commit/check_no_cpp_sources.py
uv run python tools/pre_commit/check_triton_import.py
bash scripts/check_lite_imports.sh
uv run ruff check .
uv run mypy vllm  # 待当前 519-error baseline 清零后启用为 required
```

实际脚本名以 `.pre-commit-config.yaml` 为准，CI 直接运行脚本即可，不必为两个架构
守卫安装整套 pre-commit 环境。workflow path filter 必须覆盖 hooks、CI、配置、
`AGENTS.md` 和依赖锁文件。基线清零前保留现有窄范围 mypy 检查；不得以 ignore 规则
伪造全量通过。ROCm 自托管 job 继续负责 GPU correctness 与性能门禁。

### 11.2 性能证据规则

`tests/e2e_full_benchmark.py` 只输出一次原始结果。外层 A/B 编排器用两个 worktree
交错运行至少三轮并比较中位数，保留每轮原始 JSON。

baseline fingerprint 至少包含 checkpoint/model、GPU 架构、ROCm、Torch、Triton、
profile/KV 类型和 batch/prompt/decode shape。fingerprint 不一致时可记录结果，但不得
作为回归 gate。

block-table 整行 H2D 更新是待 profile 项：仅在端到端证据成立时，改为“新增 block
时更新追加 ID”，并复用已有 decode scratch。单请求 fast path 已使用预分配 scratch；
不得把 batched 路径的问题泛化到全部请求。

异构 prompt 已按 processed `seq_len` 同批推进，不能以“prefill 基本串行”为理由重写
planner。`RLock` 也不是单 GPU 吞吐瓶颈；要解决的是事件循环同步等待锁。

FP8 KV 的默认 `k_scale/v_scale=1.0` 维持为待验证假设。P2 用多个模型和上下文长度的
FP16-KV 对照 logit parity 决定：通过则记录并保持，失败才加载或校准 scale；不在热路径
自动校准。

## 12. 分阶段实施

### P0-A：KV 与上下文安全

- 修正 block 池 `+1`。
- admission 总上下文校验和 decode 长度防线。
- 去掉 `ensure_blocks()` 静默截断。
- paged-attention 两条读路径加行容量 mask。
- 4096 限制改为显式错误或规划告警。

验收：满池、边界上下文、超限拒绝和 kernel 边界测试全部通过，再跑完整模型正确性
回归。

### P0-B：生命周期与 fatal

- backend 唯一 teardown。
- backend 执行异常不再提前 free 请求。
- stream `finally` 清理和消费者取消 abort。
- duplicate ID fail fast。
- worker 锁内同步 fatal，停止重试。
- 覆盖五路生命周期与 LoRA/KV/slot 不变量。

验收：人为注入 prefill/decode 异常后，所有消费者终止、资源归零、driver 停止、新
请求被 503 拒绝。

### P0-C：服务输入边界与托管 CI

- 限制多模态 scheme、字节、像素、timeout 和路径。
- 在 JSON 解析前限制 HTTP body；默认 loopback，debug routes 默认关闭。
- trust_remote_code 改为显式 opt-in，并 pin 远程 revision。
- 加 CPU smoke、env registry、no-C++、Triton-import 和完整 mypy 门禁。
- 同步环境变量文档。

### P1：API 与异步边界

- admission/stream 分离。
- engine 同步调用移入 worker thread。
- chat template、参数契约、usage、finish reason、disconnect abort 和 lifespan。
- server fatal/liveness/readiness 与进程监督契约；admission 后未迭代 stream 的 close。
- adapter architecture fail closed、tuning key registry 校验。
- PrefixCache 按 GPU bytes/blocks 预算；在预算实现前默认禁用，而非仅限制 entry 数。

### P2：证据驱动的性能与清理

- profile block-table 更新后决定是否增量化。
- 外层交错 A/B 和 fingerprinted baseline。
- 删除已确认不可达的 FP8 MoE 分支、`fp8_gemm.py` 及无调用方上游残留。
- 删除仅有测试调用的旧 `KVCacheAllocator`、未调用的 `_load_gguf_weights()`、
  `vllm/kv_cache_interface.py` 和 `with_cancellation()`；删除前各加一次生产调用点检查。
- 删除无引用的 GitHub workflow scripts/matchers。tokenize router 是 attach-only
  compatibility surface，只有同时移除 capability 声明与测试时才删除；model interfaces
  逐符号确认调用方，不按目录清理。
- `scripts/check_lite_imports.sh` 接入托管 CI，继续执行
  `DEPENDENCY_CLOSURE.md` 的边界，不作为无引用脚本删除。
- 修正 int4 divisor 注释为 7；不改已正确的对称量化代码。
- atomic reduce 的非确定性写入能力声明和测试门禁。
- 删除死路径后，按实际存活的 safetensors/model-construction 职责拆分
  `model_loader` 与 host policy/profile；不重写已验证 kernel。

## 13. 验证计划

### 13.1 CPU 单元与集成测试

- `RequestScheduler`：duplicate ID、queue identity、五种终态、重复 teardown。
- `LiteEngine`：LoRA add/remove 配对、queue timeout、abort、fatal admission 拒绝。
- `AsyncDriver`：error handler 与 step 不并发、fatal 后退出、不继续 sleep/retry。
- `AsyncLLM`：取消触发 abort、event loop 在 engine lock 被占时仍可运行。
- HTTP：admission 错误非 200、model 校验、unsupported 参数、disconnect cleanup。
- HTTP：debug 默认 404/禁用、超大 chunked body 在 JSON 解析前拒绝、fatal health 契约。
- 多模态：超大 base64、非法 MIME、decompression bomb、redirect、file/path 拒绝。
- Runtime planner/allocator：所有 slot 的最大上下文同时占满。

### 13.2 GPU 与数值测试

- paged attention 增加 0-token、最大 token、最后一个合法 block 和越界防御用例。
- kernel/KV 变更运行 PyTorch reference parity。
- 运行：

```bash
bash tests/run_regression_suite.sh
bash tests/run_inference_correctness_regression.sh
```

性能改动另跑至少三轮交错 A/B；正确性修复不以性能波动为由回退安全不变量。

## 14. 完成标准

本设计在以下条件全部满足时完成：

- 五种请求终态均不会挂起或泄漏 stream、slot、KV、LoRA。
- 重复 ID 在修改任何状态前失败。
- 满 block 池可服务 `max_active_requests` 个最大上下文请求。
- host 校验失效时，paged attention 读侧仍不能跨 block-table 行。
- 任何 post-admission 未预期 step 异常只处理一次并使服务进入 fatal。
- admission 错误在返回 SSE 200 前可见，断连请求停止占用 GPU。
- 默认 HTTP 服务不能读取本地文件或无界远程图片。
- 默认 server 不对外绑定、不暴露 debug endpoint，也不执行未显式许可的远程模型代码。
- PrefixCache 不能使 GPU 使用量越过规划预算；未开始迭代的 admitted stream 也能释放。
- 每个 PR 都经过托管 CPU smoke 与架构守卫；GPU job 仍执行模型正确性。
- 文档不再把废弃 env 或未实现参数描述为受支持能力。

## 15. 预期改动面

主要文件预计包括：

- `vllm/engine/request_builder.py`
- `vllm/engine/runtime_planner.py`
- `vllm/engine/block_allocator.py`
- `vllm/engine/kv_block_manager.py`
- `vllm/kernels/triton/paged_attention.py`
- `vllm/engine/request_scheduler.py`
- `vllm/engine/backend/lite_single_gpu.py`
- `vllm/engine/lite_engine.py`
- `vllm/engine/async_driver.py`
- `vllm/engine/async_llm.py`
- `vllm/engine/multimodal_processor.py`
- `vllm/entrypoints/openai/api_server.py`
- `.github/workflows/` 与对应 CPU 测试

每个 P0 子阶段应独立提交和验证。不要把正确性修复、API 扩面、性能实验和大规模死
代码删除混在同一个变更中。
