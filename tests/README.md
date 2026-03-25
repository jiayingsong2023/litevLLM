# 测试与回归套件

本目录保留 **推理准确度**、**性能**、**API/smoke**、**LoRA / 多模态** 相关入口；与 HuggingFace 数值对拍、GGUF 审计等**手工工具**在 [`tests/tools/`](tools/README.md)。

更完整的档位说明见 **[`docs/INFERENCE_ACCURACY.md`](../docs/INFERENCE_ACCURACY.md)**。

## 三模型推理准确度回归（TinyLlama + Qwen3.5 9B AWQ + Qwen3.5 35B AWQ）

在仓库根目录、本地已放置对应 `models/...` 且 GPU 驱动正常时：

```bash
FASTINFERENCE_KV_FP8=0 bash tests/run_inference_accuracy_regression.sh
```

仅跑 B 档（更快）：`SKIP_A_TIER=1 bash tests/run_inference_accuracy_regression.sh`  
跳过 35B（本机资源不足时）：`SKIP_35B=1 bash tests/run_inference_accuracy_regression.sh`
附带性能诊断（prefill/decode + AWQ fallback 计数）：`RUN_PERF_DIAG=1 bash tests/run_inference_accuracy_regression.sh`

默认模型路径可通过环境变量覆盖：

- `MODEL_TINYLLAMA`
- `MODEL_QWEN35_9B_AWQ`
- `MODEL_QWEN35_35B_AWQ`
- `HF_QWEN35_9B_FP16`
- `HF_QWEN35_35B_FP16`

脚本会对 `Qwen3.5-35B-AWQ` 自动启用稳定运行配置（FP8/KV + MoE offload 相关环境变量），并在缺少 `HF_QWEN35_35B_FP16` 时自动跳过 35B 的 A-tier 对拍。

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
| Qwen GGUF 张量审计 | `tests/tools/qwen35_gguf_alignment_audit.py` || 性能回归（TinyLlama + Qwen3.5 9B/35B AWQ） | `uv run python tests/e2e_full_benchmark.py --models tinyllama,qwen35_9b_awq,qwen35_35b_awq --json-out .tmp_perf_regression_awq.json` |
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
