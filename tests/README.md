# 测试与回归套件

本目录保留 **推理准确度**、**性能**、**API/smoke**、**LoRA / 多模态** 相关入口；与 HuggingFace 数值对拍、GGUF 审计等**手工工具**在 [`tests/tools/`](tools/README.md)。

更完整的档位说明见 **[`docs/INFERENCE_ACCURACY.md`](../docs/INFERENCE_ACCURACY.md)**。

## 四模型推理准确度回归（TinyLlama + Qwen3.5 9B AWQ/GGUF + DeepSeek-V2-Lite GGUF）

在仓库根目录、本地已放置对应 `models/...` 且 GPU 驱动正常时：

```bash
FASTINFERENCE_KV_FP8=0 bash tests/run_inference_accuracy_regression.sh
```

仅跑 B 档（更快）：`SKIP_A_TIER=1 bash tests/run_inference_accuracy_regression.sh`

A 档中 DeepSeek 为两步：**A4a** 同一 `DeepSeek-V2-Lite-Chat` safetensors 上 Lite vs HF（`--regression-gate safetensors`，与 transformers 实现对齐）；**A4b** `DeepSeek-V2-Lite-GGUF` vs Chat bf16（`--regression-gate deepseek-gguf`，Q4 与 bf16  logits 不要求与 A4a 同等 CosSim）。可调环境变量：`FASTINFERENCE_DEEPSEEK_GGUF_REGRESSION_MIN_COS`、`FASTINFERENCE_DEEPSEEK_GGUF_REGRESSION_MIN_TOPK`。

## 一键 pytest（默认，不加载整模）

```bash
cd /path/to/FastInference && bash tests/run_regression_suite.sh
```

等价于对下列文件的 `pytest`：`test_qwen35_*`、`test_moe_gguf_packed`、`test_quality_bar_spotcheck_heuristics`、`test_logits_dump_stats`、`test_lora_registry_smoke`、`test_multimodal_registry_smoke`、`lite_smoke_test`。

## 模型覆盖矩阵（本地需有 `models/...` 目录）

| 场景 | 入口 |
|------|------|
| B 档观感（Qwen / DeepSeek / GLM / 35B 等） | `uv run python tests/tools/quality_bar_spotcheck.py`（见文档 §5） |
| A 档 logits（HF vs Lite） | `tests/verify_semantic_integrity.py`、`tests/verify_layer0_submodule_alignment.py`、`tests/verify_layerwise_alignment.py` |
| DeepSeek 末位 logits / 逐层 hidden | `tests/tools/compare_hf_lite_deepseek_logits.py`（A 档宜 **同一 safetensors 目录**；GGUF 对 HF bf16 不做 CosSim≥0.99 要求）、`compare_hf_lite_deepseek_layer_hiddens.py` |
| Qwen 35B MoE GGUF packed vs dense | `tests/tools/qwen35_moe_packed_lite_logits_audit.py` |
| Qwen GGUF 张量审计 | `tests/tools/qwen35_gguf_alignment_audit.py` |
| 性能（多模型 AsyncLLM） | `uv run python tests/e2e_full_benchmark.py` |
| 算子 / 量化微基准（合成权重） | `uv run python tests/full_perf_regression.py` |
| LLM.generate 冒烟（与 e2e 模型列表对齐） | `uv run python tests/test_offline_api.py` |

## 整模验证注意事项

`verify_semantic_integrity.py` 等会加载完整权重与 KV，显存不足或路径错误时可能长时间占用 GPU。建议：

```bash
timeout 20m uv run python tests/verify_semantic_integrity.py --model ... --preset ...
```

卡住时（本机终端）：`pkill -9 -f 'verify_semantic_integrity.py'`。

## 已移除的测试

以下脚本已从仓库删除（与当前保留的回归目标重复或过于细碎）：`test_cuda_alloc.py`、`test_legacy_linear_attn_norm_load.py`、`test_lite_config_rope_parameters.py`、`test_operator_accuracy.py`、`verify_kernel_parity.py`、`test_async_streaming.py`、`test_verify_semantic_prefill_helpers.py`。

`LiteConfig` 的 `rope_parameters` 行为说明见 `vllm/model_executor/models/lite_config.py`。
