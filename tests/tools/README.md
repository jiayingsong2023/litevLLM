# `tests/tools/` 抽检与对齐工具

## 是否需要提交到 Git（GitHub）

**建议提交。** 本目录下的脚本属于仓库的**维护与质量工具**（推理观感抽检、与 Hugging Face 对齐、GGUF 审计等），纳入版本控制可以：

- 让其他贡献者与 CI 用**同一条命令**复现检查；
- 与 `docs/INFERENCE_ACCURACY.md`、`tests/` 形成固定工作流。

**可以不提交的情况**：仅个人调试用的**一次性**脚本、含本机绝对路径或密钥的草稿；这类更适合放在个人分支或本地，不要进主分支。

---

## 环境约定

- 在**仓库根目录**执行；Python 脚本需 **`PYTHONPATH=.`**（或使用下文 `uv run` 示例）。
- 依赖与虚拟环境以项目 **`uv`** / `pyproject.toml` 为准：`uv run python tests/tools/<name>.py`。
- 多数脚本需要本地模型目录（如 `models/...`），路径由参数或环境变量指定。

更完整的验收档位与命令索引见 **[`docs/INFERENCE_ACCURACY.md`](../docs/INFERENCE_ACCURACY.md)**。

---

## 脚本一览

| 脚本 | 作用 |
|------|------|
| [`quality_bar_spotcheck.py`](quality_bar_spotcheck.py) | **B 档**：`LiteEngine` 固定 prompt 续写，可读性/连贯性等启发式粗筛（不对比 HF）。 |
| [`run_inference_quality_suite.sh`](run_inference_quality_suite.sh) | 一键跑 B 档 greedy（可扩展 `RUN_A_TIER`）；封装 `quality_bar_spotcheck.py`。 |
| [`report_expected_alignment_metrics.py`](report_expected_alignment_metrics.py) | 汇总单元测试中的对齐预期（见文档 §4）。 |
| [`qwen35_gguf_alignment_audit.py`](qwen35_gguf_alignment_audit.py) | GGUF 反量化权重与 HF `safetensors` 逐张量对比（含可选 `--conv-check`）。 |
| [`qwen35_gated_delta_conv_alignment.py`](qwen35_gated_delta_conv_alignment.py) | 在权重一致时，检查 GatedDeltaNet 前向中 conv+SiLU 与 HF 对齐。 |
| [`qwen35_chunk_gated_delta_alignment.py`](qwen35_chunk_gated_delta_alignment.py) | Chunk 规则与 FLA naive 等参考实现对比。 |
| [`verify_qwen35_final_hidden_alignment.py`](verify_qwen35_final_hidden_alignment.py) | 最后一层 / 逐层 hidden 与 HF 的 CosSim（可选 `--per-layer`）。 |
| [`qwen35_moe_packed_lite_logits_audit.py`](qwen35_moe_packed_lite_logits_audit.py) | 35B MoE GGUF：`dump` / `diff` / `weight-parity`（packed vs dense）。 |
| [`compare_hf_lite_deepseek_logits.py`](compare_hf_lite_deepseek_logits.py)、[`compare_hf_lite_deepseek_layer_hiddens.py`](compare_hf_lite_deepseek_layer_hiddens.py) | DeepSeek-V2-Lite：HF vs Lite 末位 logits / 逐层 hidden。 |
| [`diagnose_deepseek_hf_lite_logits.py`](diagnose_deepseek_hf_lite_logits.py) | DeepSeek 分层诊断（指向上述 compare 脚本）。 |

---

## 快速示例

```bash
# B 档（观感，贪婪）
PYTHONPATH=. uv run python tests/tools/quality_bar_spotcheck.py \
  --model models/Qwen3.5-9B-FP16 --quant none --prompt-subset minimal --temperature 0

# 一键 shell（B 档）
MODEL=models/Qwen3.5-9B-FP16 bash tests/tools/run_inference_quality_suite.sh

# GGUF 权重审计
PYTHONPATH=. uv run python tests/tools/qwen35_gguf_alignment_audit.py \
  --gguf /path/to/model.gguf --hf-dir /path/to/FP16 --conv-check
```

---

## 与 `tests/` 的区别

- **`tests/tools/`**：偏**人工或本地驱动**的抽检、审计、对齐脚本，参数灵活，模型路径常由用户指定。
- **`tests/`**：**pytest** 等自动化测试（如 **`verify_semantic_integrity.py`** 做 **A 档** HF 对比），适合 CI；与 `tests/tools/` 互补，见 [`tests/README.md`](../tests/README.md)。
