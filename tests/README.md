# 测试与回归套件

本目录包含 **推理准确度**、**性能**、**API/smoke** 相关入口。当前主要支持 **Gemma4** (26B/31B) 及 **Qwen3.5-9B** 系列模型。

更完整的档位说明见 **[`docs/INFERENCE_ACCURACY.md`](../docs/INFERENCE_ACCURACY.md)**。

## 核心回归测试 (Gemma4 + Qwen3.5)

在仓库根目录、本地已放置对应 `models/...` 且 GPU 驱动正常时：

```bash
uv run bash tests/run_inference_correctness_regression.sh
```

（默认 `FASTINFERENCE_KV_TYPE=turbo_int4`；可显式设置 `FASTINFERENCE_KV_TYPE=fp16|fp8|turbo_int4`。）

仅跑 B 档（更快）：`SKIP_A_TIER=1 bash tests/run_inference_correctness_regression.sh`  
附带性能诊断：`RUN_PERF_DIAG=1 bash tests/run_inference_correctness_regression.sh`

## Gemma4 专项审计与调试

Gemma4 系列（26B/31B）具备专用的 Spotcheck 门控与审计：

- **Gemma4-26B A4B (`models/gemma-4-26B-A4B-it-AWQ-4bit`)**:
  - `tests/tools/quality_bar_spotcheck.py` 自动启用专属门控逻辑。
  - 检查包含：结构可读性（拦截乱码）与语义锚点（Prompt 语义对齐）。
- **Gemma4 A-strict (prefill-only)**: 
  - `tests/tools/gemma4_prefill_strict_audit.py`
  - 使用 sequential GPU reference 进行 HF 对拍。
- **Gemma4 Long Decode 诊断**:
  - `tests/tools/gemma4_layer_drift_diagnostic.py` (分析 local/full 层在长序列下的 drift)。

## 性能基准测试

- **性能回归测试**: 
  ```bash
  uv run python tests/e2e_full_benchmark.py --models gemma4_26b_a4b,gemma4_31b_q4 --gemma26b-concurrent 1 --gemma31b-concurrent 1
  ```

## 关键测试文件索引

| 功能 | 测试/工具入口 |
|------|------|
| 核心正确性回归 | `tests/run_inference_correctness_regression.sh` |
| Gemma4 语义审计 | `tests/verify_semantic_integrity.py` |
| 性能基准测试 | `tests/e2e_full_benchmark.py` |
| 质量门控审计 | `tests/tools/quality_bar_spotcheck.py` |
| 预解量化缓存测试 | `tests/test_e2e_warmup_config.py` |

---
*注意：`LiteConfig` 的 `rope_parameters` 行为说明见 `vllm/model_executor/models/lite_config.py`。*
