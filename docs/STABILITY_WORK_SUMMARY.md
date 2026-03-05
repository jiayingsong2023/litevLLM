# 稳定性修复与回归保障（简要说明）

本文档用于简要记录本轮稳定性工作，目标是避免再次出现“代码可导入但服务不可运行”的问题。

## 1. 本轮目标

- 修复 P0 级别阻断问题（入口导入失败、模块缺失、能力声明不一致）。
- 打通 `openai + tokenize + pooling` 关键导入链与基础服务路径。
- 增加可执行的 smoke 测试（含 HTTP 端到端）。
- 将 smoke 接入 CI，形成提交后自动回归机制。

## 2. 已完成工作

### P0（可运行性修复）

- 修复兼容入口：
  - `vllm/entrypoints/api_server.py`
  - 统一转发到 `vllm.entrypoints.openai.api_server`
- 修复/重建核心 serving 基础模块：
  - `vllm/entrypoints/openai/engine/serving.py`
  - `vllm/entrypoints/openai/models/serving.py`
- 补齐缺失符号与依赖链（典型如工具调用 ID、异步工具、类型兼容类等）。
- 量化能力声明对齐：
  - 去除 `gptq` 的“误支持”状态，明确 Lite 构建下不可用。

### P1（路径一致性修复）

- 修复 `serve/tokenize`、`pooling/*` 路径中的导入/语法断点。
- 补齐轻量兼容实现，确保关键模块可导入、可被 smoke 覆盖。
- 增加服务路径一致性测试，验证旧入口与新入口行为连接正常。

### P2（端到端与真实模型轻量回归）

- 新增最小 HTTP smoke，覆盖：
  - `POST /tokenize`
  - `POST /detokenize`
  - `POST /v1/chat/completions`
- 新增真实模型轻量回归：
  - 有模型时执行 `AutoConfig` + `AutoTokenizer` 加载与 encode/decode roundtrip
  - 无模型时自动 skip，避免无意义失败

## 3. 新增 smoke 测试文件

- `tests/smoke/test_entrypoint_imports.py`
- `tests/smoke/test_serving_path_consistency.py`
- `tests/smoke/test_http_endpoints_smoke.py`
- `tests/smoke/test_real_model_light_regression.py`

## 4. CI 接入

新增工作流：

- `.github/workflows/smoke.yml`

执行流程：

1. `actions/setup-python`（3.12）
2. `pip install uv`
3. `uv sync`
4. `uv run pytest -q tests/smoke`

触发范围：

- `vllm/**/*.py`
- `tests/smoke/**/*.py`
- `pyproject.toml`
- `uv.lock`
- 工作流文件本身

## 5. 如何本地复现

```bash
uv sync
uv run pytest -q tests/smoke
```

## 6. 后续建议（可选）

- 增加一个更快的“前置导入检查”任务（仅做关键模块 `py_compile/import`）。
- 逐步补功能级回归（例如 streaming chunk 格式、tokenize 参数边界）。
- 对当前保留的 Lite 兼容实现做分层收敛，减少“临时兼容代码”长期膨胀风险。
