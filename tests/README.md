# 测试与回归套件

本目录包含 **推理准确性回归**、**性能基准**、**专项诊断** 三类入口。当前默认回归覆盖：

- `TinyLlama-1.1B-Chat-v1.0`
- `Qwen3.5-9B-AWQ`
- `Gemma4-31B-it-AWQ-4bit`
- `Gemma4-26B-A4B-it-AWQ-4bit`

更完整的档位说明见 [docs/INFERENCE_ACCURACY.md](../docs/INFERENCE_ACCURACY.md)。

## 主回归入口

准确性主入口：

```bash
bash tests/run_inference_correctness_regression.sh
```

默认行为：

- 先跑 Tier-B `quality_bar_spotcheck`
- `TinyLlama` 和 `Qwen3.5-9B-AWQ` 继续跑 A-strict
- `Gemma4-31B` 默认跑 Tier-B + A-lite
- `Gemma4-26B` 默认跑 Tier-B + A-lite；`A-strict` 默认开启但本机可按需关闭

常用选项：

```bash
SKIP_A_TIER=1 bash tests/run_inference_correctness_regression.sh
RUN_PERF_DIAG=1 bash tests/run_inference_correctness_regression.sh
RUN_GEMMA4_A_STRICT=0 RUN_GEMMA4_26B_A_STRICT=0 bash tests/run_inference_correctness_regression.sh
```

路径与 prompts 默认值：

- `MODEL_TINYLLAMA=models/TinyLlama-1.1B-Chat-v1.0`
- `MODEL_QWEN35_9B_AWQ=models/Qwen3.5-9B-AWQ`
- `MODEL_GEMMA4_31B_Q4=models/gemma-4-31B-it-AWQ-4bit`
- `MODEL_GEMMA4_26B_A4B=models/gemma-4-26B-A4B-it-AWQ-4bit`
- `TINYLLAMA_PROMPTS_FILE=tests/tools/fixtures/tinyllama_correctness_prompts_default.json`
- `GEMMA4_PROMPTS_FILE=tests/tools/fixtures/gemma4_correctness_prompts_default.json`

说明：

- TinyLlama 默认 prompt 集已经移除已知不稳定的 `zh_gd` 用例，避免它阻塞后续 Gemma4 默认路径。
- Gemma4 默认走当前已验收推荐 profile：
  - `31B`: `FASTINFERENCE_GEMMA4_DENSE_DOWN_PROJ=1` + `FASTINFERENCE_AWQ_DECODE_GEMV=1` + `FASTINFERENCE_AWQ_GROUP32_GEMV_ALL=1` + `FASTINFERENCE_AWQ_FUSED_GATE_UP=1`
  - `26B`: `FASTINFERENCE_AWQ_DECODE_GEMV=1` + `FASTINFERENCE_AWQ_FUSED_GATE_UP=1`
  - 两者都默认带 `FASTINFERENCE_GPU_GREEDY_*` 和 `FASTINFERENCE_KV_MAX_*` 限制

## 性能基准入口

性能主入口：

```bash
uv run python tests/e2e_full_benchmark.py
```

Gemma4 常用命令：

```bash
uv run python tests/e2e_full_benchmark.py \
  --models gemma4_26b_a4b,gemma4_31b_q4 \
  --gemma26b-concurrent 1 \
  --gemma31b-concurrent 1
```

当前行为：

- `gemma4_31b_q4` 和 `gemma4_26b_a4b` 的 `stable_env` 已内置当前推荐 profile
- 在 ROCm 上，如果一次运行里包含多个 Gemma4 大模型，脚本会默认启用 **per-model process isolation**
- 这个隔离模式用于避免顺序大模型 benchmark 时的显存残留 / allocator fragmentation 导致的 OOM

可显式控制：

```bash
uv run python tests/e2e_full_benchmark.py --model-process-isolation ...
uv run python tests/e2e_full_benchmark.py --no-model-process-isolation ...
```

常用参数：

- `--models tinyllama,qwen35_9b_awq,gemma4_31b_q4,gemma4_26b_a4b`
- `--gemma31b-concurrent 1`
- `--gemma26b-concurrent 1`
- `--gemma31b-prompt-tokens N`
- `--gemma26b-prompt-tokens N`
- `--warmup-preset default|off|cold`
- `--json-out <path>`

## Gemma4 专项工具

- Tier-B 门控：`tests/tools/quality_bar_spotcheck.py`
- A-lite smoke：`tests/tools/gemma4_single_prompt_smoke.py`
- A-strict prefill audit：`tests/tools/gemma4_prefill_strict_audit.py`
- Layer drift / 长 decode 诊断：`tests/tools/gemma4_layer_drift_diagnostic.py`

本机当前验收策略：

- `Gemma4-31B/26B` 必跑：A-lite + Tier-B
- `strict audit` 不作为本机阻塞项，除非专项排查

## 关键文件索引

| 功能 | 入口 |
|------|------|
| 主正确性回归 | `tests/run_inference_correctness_regression.sh` |
| 主性能基准 | `tests/e2e_full_benchmark.py` |
| Tier-B 质量门控 | `tests/tools/quality_bar_spotcheck.py` |
| Gemma4 A-lite | `tests/tools/gemma4_single_prompt_smoke.py` |
| Gemma4 A-strict | `tests/tools/gemma4_prefill_strict_audit.py` |
| 语义一致性校验 | `tests/verify_semantic_integrity.py` |

---

注意：`LiteConfig` 的 `rope_parameters` 行为说明见 `vllm/model_executor/models/lite_config.py`。
