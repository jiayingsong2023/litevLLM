# FastInference API 参考（Lite）

本文档面向当前 Lite 路线，汇总项目中的主要 HTTP API、最小请求字段和示例。

## 1. 启动服务

```bash
uv run python -m vllm.entrypoints.openai.api_server \
  --model models/Qwen3.5-9B-GGUF \
  --policy-mode auto \
  --host 0.0.0.0 \
  --port 8000
```

默认服务入口：`http://127.0.0.1:8000`

策略参数说明：

- `--policy-mode auto|aggressive|stable`
  - `auto`: 启动时根据显存与模型规模一次性选择策略（推荐）
  - `aggressive`: 高并发高吞吐（小模型优先）
  - `stable`: 低并发/串行优先（大模型可用性优先）
- 也可通过环境变量 `FASTINF_POLICY_MODE` 设置默认值。

Qwen3.5 额外策略开关（固定策略，不做运行中动态降档）：

- `FASTINFERENCE_QWEN9_AGGRESSIVE`（默认 `1`）
  - 仅作用于 `Qwen3.5-9B Dense` 路径
  - `1`: 激进高吞吐
  - `0`: 关闭激进
- `FASTINFERENCE_QWEN9_STABLE`（默认 `0`）
  - `1`: 强制 9B 稳定策略（优先级高于 `FASTINFERENCE_QWEN9_AGGRESSIVE`）
- `FASTINFERENCE_QWEN35_FP32_LINEAR`（默认 `0`，MoE 场景由加载器稳定策略控制）
  - 用于显式覆盖 35B 线性层计算精度策略
- `FASTINFERENCE_QWEN35_GROUPED_MOE`（默认 `1`）
  - 35B MoE expert grouped 批量化开关
- `FASTINFERENCE_QWEN35_GROUPED_MOE_MIN_TOKENS`（默认 `2`）
  - 仅在 token 数达到阈值时启用 grouped（`BS=1` 建议保持旧快路径）

示例：

```bash
# 9B 默认激进（可省略 FASTINFERENCE_QWEN9_AGGRESSIVE）
uv run python -m vllm.entrypoints.openai.api_server --model models/Qwen3.5-9B-GGUF

# 9B 强制稳定
FASTINFERENCE_QWEN9_STABLE=1 \
uv run python -m vllm.entrypoints.openai.api_server --model models/Qwen3.5-9B-GGUF

# 35B MoE grouped 打开且阈值=2（推荐）
FASTINFERENCE_QWEN35_GROUPED_MOE=1 FASTINFERENCE_QWEN35_GROUPED_MOE_MIN_TOKENS=2 \
uv run python -m vllm.entrypoints.openai.api_server --model models/Qwen3.5-35B-MoE-GGUF
```

## 2. 默认可用 API（openai/api_server）

### GET `/v1/models`

- 用途：查看当前加载模型
- 响应：OpenAI 风格 `list`，`data[0].id` 为模型标识

示例：

```bash
curl -s http://127.0.0.1:8000/v1/models
```

### POST `/v1/chat/completions`

- 用途：聊天补全
- 最小请求字段：
  - `model: string`
  - `messages: [{role, content}]`
- 常用可选字段：
  - `stream: bool`（默认 `false`）
  - `max_tokens: int`（默认 `128`）
  - `temperature: float`（默认 `0.7`）

非流式示例：

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "models/Qwen3.5-9B-GGUF",
    "messages": [{"role":"user","content":"你好，请用一句话介绍你自己。"}],
    "stream": false,
    "max_tokens": 64
  }'
```

流式示例（SSE）：

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "models/Qwen3.5-9B-GGUF",
    "messages": [{"role":"user","content":"给我一个三步计划。"}],
    "stream": true
  }'
```

## 3. 扩展 API（tokenize / pooling / embedding / classify / score）

以下接口在 `vllm.entrypoints` 中已实现路由与协议，通常由更完整的服务装配流程挂载。

### Tokenize

- `POST /tokenize`
  - 请求：`{ model?, prompt }` 或 `chat messages` 形式
  - 响应：`{ count, max_model_len, tokens, token_strs? }`
- `POST /detokenize`
  - 请求：`{ model?, tokens: number[] }`
  - 响应：`{ prompt }`

### Pooling / Embedding / Classify / Score

- `POST /pooling`
- `POST /v1/embeddings`
- `POST /classify`
- `POST /score`
- 兼容别名：
  - `POST /v1/score`
  - `POST /rerank`
  - `POST /v1/rerank`
  - `POST /v2/rerank`

这些接口的请求模型定义可参考：

- `vllm/entrypoints/pooling/pooling/protocol.py`
- `vllm/entrypoints/pooling/embed/protocol.py`
- `vllm/entrypoints/pooling/classify/protocol.py`
- `vllm/entrypoints/pooling/score/protocol.py`
- `vllm/entrypoints/serve/tokenize/protocol.py`

## 4. 错误响应格式

统一错误结构（摘要）：

```json
{
  "error": {
    "message": "error detail",
    "type": "BadRequestError",
    "param": null,
    "code": 400
  }
}
```

## 5. 回归建议

建议每次 API 相关变更后执行：

```bash
uv run pytest -q tests/smoke
```

其中已经覆盖了：

- 关键模块导入回归
- 服务路径一致性回归
- HTTP smoke（`/tokenize`、`/detokenize`、`/v1/chat/completions`）
- 真实模型轻量加载回归
