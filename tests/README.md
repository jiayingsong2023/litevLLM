# 测试与回归套件

本目录保留 **推理准确度**、**性能**、**API/smoke**、**LoRA / 多模态** 相关入口；与 HuggingFace 数值对拍、GGUF 审计等**手工工具**在 [`tests/tools/`](tools/README.md)。当前默认测试面以 `lite-only` 主线为准，不再把 `Qwen3.5-35B` 视为正式回归目标。

更完整的档位说明见 **[`docs/INFERENCE_ACCURACY.md`](../docs/INFERENCE_ACCURACY.md)**。

## 默认推理准确度回归（TinyLlama + Qwen3.5 9B AWQ + Gemma4 31B Q4）

在仓库根目录、本地已放置对应 `models/...` 且 GPU 驱动正常时：

```bash
bash tests/run_inference_correctness_regression.sh
```

（默认 `FASTINFERENCE_KV_TYPE=turbo_int4`；如需切换可显式设置 `FASTINFERENCE_KV_TYPE=fp16|fp8|turbo_int4`。）

仅跑 B 档（更快）：`SKIP_A_TIER=1 bash tests/run_inference_correctness_regression.sh`  
附带性能诊断（prefill/decode + AWQ fallback 计数）：`RUN_PERF_DIAG=1 bash tests/run_inference_correctness_regression.sh`

默认策略：

- `<=14B` 模型：`A-strict + B`
- `>14B` 模型：`A-lite + B`

当前脚本中：

- `TinyLlama-1.1B`、`Qwen3.5-9B-AWQ` 走 `A-strict + B`
- `Gemma4-31B-it-AWQ-4bit` 默认走 `A-lite + B`（可手动加 `A-strict`）
- `Gemma4-26B-A4B-it-AWQ-4bit` 默认走 `A-strict + A-lite + B`（text-only）
- 如需对 Gemma4-31B 手动开启严格 HF prefill 对拍，可设 `RUN_GEMMA4_A_TIER=1` 或 `RUN_GEMMA4_A_STRICT=1`

Gemma4 `A-strict`（31B/26B）的执行方式与小模型不同：

- 默认入口：`tests/tools/gemma4_prefill_strict_audit.py`
- 默认使用 `--hf-device cuda`
- 采用 **sequential GPU reference**，不是 Lite 与 HF 同时驻留 GPU
- 具体顺序是：先抓 Lite prefill logits，再释放 LiteEngine GPU 占用，再在同一张 GPU 上加载 Gemma4 reference 做一次 `prefill-only` 对拍
- 对 31B：默认仍建议 `A-lite + B`，`A-strict` 作为手动专项审计
- 对 26B-A4B：默认开启 `A-strict`（sequential GPU reference）

默认模型路径可通过环境变量覆盖：

- `MODEL_TINYLLAMA`
- `MODEL_QWEN35_9B_AWQ`
- `MODEL_GEMMA4_31B_Q4`
- `HF_QWEN35_9B_FP16`

Gemma4 默认优先探测本地目录：
`models/gemma-4-31B-it-AWQ-4bit`、`models/Gemma-4-31B-Q4`、`models/Gemma-4-31B-AWQ`、`models/Gemma-4-31B-AWQ-4bit`。
仅在这些目录都不存在时，才回退到 `cyankiwi/gemma-4-31B-it-AWQ-4bit`。

Gemma4 默认 correctness prompt 集：
`tests/tools/fixtures/gemma4_correctness_prompts_default.json`

Gemma4 边界题 / 长尾退化调试 prompt 集：
`tests/tools/fixtures/gemma4_edge_prompts_debug.json`

Gemma4-26B A4B（`models/gemma-4-26B-A4B-it-AWQ-4bit`）的 Tier-B 说明：

- `tests/tools/quality_bar_spotcheck.py` 对该模型自动启用 `gemma4_26b` hard gate（模型专用）。
- hard gate 包含两类检查：
  - 结构可读性硬规则：拦截 token-soup、短词循环、混脚本胶水词等典型乱码。
  - 固定 prompt 语义锚点：对 `gemma4_correctness_prompts_default.json` 的 `gemma_*` prompt id 校验最小语义锚（如 `capital -> paris/巴黎`、`bst -> tree+node/left/right/binary`）。
- 该机制目标是避免 Tier-B 启发式假阳性（“看似通过但文本不可读”）；它是 Gemma4-26B 定向门禁，不是通用语义评分器。

## 一键 pytest（默认，不加载整模）

```bash
cd /path/to/FastInference && bash tests/run_regression_suite.sh
```

当前默认集合包含：

- `tests/test_kv_default_policy.py`
- `tests/test_quality_bar_spotcheck_heuristics.py`
- `tests/test_e2e_warmup_config.py`
- `tests/test_logits_dump_stats.py`
- `tests/lite_smoke_test.py`
- `tests/test_gemma4_strict_audit_smoke.py`
- `tests/test_gemma4_26b_strict_warn_only.py`
- `tests/test_model_registry_gemma4.py`
- `tests/test_gemma4_reference_loader.py`
- `tests/test_gemma4_diagnostics_warn_only.py`
- `tests/test_run_inference_correctness_regression.py`

其中 Gemma4 相关默认覆盖四类保护（含一个非 gate 诊断入口）：

- `test_gemma4_strict_audit_smoke.py`：校验 `tests/tools/gemma4_prefill_strict_audit.py` 的参数解析、失败快返、命令与环境变量拼装
- `test_model_registry_gemma4.py`：校验 `vllm/model_executor/models/registry.py` 在 `architectures=[]` 时对 Gemma4 的推断与解析（避免 `Unsupported architectures: []` 回归）
- `test_gemma4_reference_loader.py`：校验 `tests/verify_semantic_integrity.py` 中 Gemma4 reference loader 的关键映射与赋值逻辑（`weight_packed/scale/shape`、vision/audio 忽略、零赋值失败）
- `test_gemma4_diagnostics_warn_only.py`：默认仅校验 strict/drift 输出解析与基线结构；设 `RUN_GEMMA4_DIAGNOSTIC=1` 时会实际运行 Gemma4 A-strict + long-drift，并对比基线，差异只发 warning 不作为失败门槛
- `test_run_inference_correctness_regression.py`：校验 `RUN_GEMMA4_A_STRICT=1` 时，`tests/run_inference_correctness_regression.sh` 会实际触发 Gemma4 A-strict 入口

Gemma4 诊断基线文件：

- `tests/tools/fixtures/gemma4_a_strict_baseline.json`（CosSim + argmax）
- `tests/tools/fixtures/gemma4_26b_a_strict_baseline.json`（26B A-strict CosSim + argmax）
- `tests/tools/fixtures/gemma4_layer_drift_baseline_short_hi.json`（token 16/24/32 层漂移摘要）
- 一键复检入口：`bash tests/run_gemma4_diagnostics_warn_only.sh`（可附加 pytest 参数，例如 `-k parse_helpers`）
- Gemma4-26B 一键 strict 复检入口：`bash tests/run_gemma4_26b_diagnostics_warn_only.sh`

## 模型覆盖矩阵（本地需有 `models/...` 目录）

| 场景 | 入口 |
|------|------|
| B 档观感（Qwen / DeepSeek / GLM 等） | `uv run python tests/tools/quality_bar_spotcheck.py`（见文档 §5） |
| A-strict（HF vs Lite） | `tests/verify_semantic_integrity.py`、`tests/verify_layer0_submodule_alignment.py`、`tests/verify_layerwise_alignment.py` |
| Gemma4 A-strict（prefill-only, sequential GPU reference） | `tests/tools/gemma4_prefill_strict_audit.py` |
| A-lite（关键点审计） | `tests/tools/gemma4_single_prompt_smoke.py`（固定 2 到 3 个 prompt，检查首 token / 正常结束 / 文本完整性） |
| Gemma4 long decode 诊断 | `tests/tools/gemma4_layer_drift_diagnostic.py`（默认 `short_hi`，输出 local/full 层在 16/24/32 token 的 drift 摘要） |
| DeepSeek 末位 logits / 逐层 hidden | `tests/tools/compare_hf_lite_deepseek_logits.py`（A 档宜 **同一 safetensors 目录**；GGUF 对 HF bf16 不做 CosSim≥0.99 要求）、`compare_hf_lite_deepseek_layer_hiddens.py` |
| Qwen GGUF 张量审计 | `tests/tools/qwen35_gguf_alignment_audit.py` |
| 性能回归（TinyLlama + Qwen3.5 9B AWQ + Gemma4 31B Q4） | `uv run python tests/e2e_full_benchmark.py --models tinyllama,qwen35_9b_awq,gemma4_31b_q4 --json-out .tmp_perf_regression_awq.json` |
| AWQ fused GEMM 微基准（可选） | `uv run python tests/bench_awq_fused_gemm_ab.py` |
| LLM.generate 冒烟（与 e2e 模型列表对齐） | `uv run python tests/test_offline_api.py` |
| 参数网格自动调优（批量跑 + 自动选优 + 归档） | `uv run python tests/tools/perf_grid_search.py --model-key gemma4_26b_a4b --phase full --run-correctness-on-best --correctness-skip-a-tier` |

Day5 warmup / 首请求抖动治理（E2E）可通过下列参数或同名环境变量控制：

- `--warmup-prefill-rounds`
- `--warmup-decode-rounds`
- `--warmup-decode-tokens`
- `--warmup-burst-rounds`
- `--warmup-burst-concurrency`
- `--warmup-burst-decode-tokens`

## 整模验证注意事项

`verify_semantic_integrity.py` 等会加载完整权重与 KV，显存不足或路径错误时可能长时间占用 GPU。建议：

```bash
timeout 20m uv run python tests/verify_semantic_integrity.py --model ... --preset ...
```

卡住时（本机终端）：`pkill -9 -f 'verify_semantic_integrity.py'`。

## 已移除的测试

以下脚本已从仓库删除（与当前保留的回归目标重复或过于细碎）：`test_cuda_alloc.py`、`test_legacy_linear_attn_norm_load.py`、`test_lite_config_rope_parameters.py`、`test_operator_accuracy.py`、`verify_kernel_parity.py`、`test_async_streaming.py`、`test_verify_semantic_prefill_helpers.py`。

`LiteConfig` 的 `rope_parameters` 行为说明见 `vllm/model_executor/models/lite_config.py`。
