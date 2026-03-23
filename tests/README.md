# Tests

日常「续写是否可读」的抽检入口见 **[`docs/INFERENCE_ACCURACY.md`](../docs/INFERENCE_ACCURACY.md)** 与 **`scripts/quality_bar_spotcheck.py`**（不对比 HF）。本目录中的 **`verify_semantic_integrity.py`** 偏 **A 档**（与 HF 数值对比），用于内核与加载器回归；两档关系与命令见该文档 **§1、§5**。

**Greedy + minimal Tier-B（固定命令，与采样噪声脱钩）**：

```bash
cd /path/to/FastInference && FASTINFERENCE_KV_FP8=0 uv run python scripts/quality_bar_spotcheck.py \
  --model models/Qwen3.5-9B-FP16 --quant none --prompt-subset minimal --temperature 0 --frugal
```

大 MoE GGUF 可再加 `FASTINFERENCE_QWEN35_MOE_PACKED_GGUF=1` 与 `--model models/Qwen3.5-35B-MoE-GGUF --quant gguf`。

## 卡住的整模验证进程（`verify_semantic_integrity.py`）

该脚本会加载完整模型 + KV，若路径错误、显存不足或引擎死循环，进程可能长时间占满 CPU/GPU。在**本机**终端执行：

```bash
# 结束所有正在跑的语义验证（含 uv 子进程）
pkill -9 -f 'verify_semantic_integrity.py'
```

建议用 `timeout` 限制墙钟时间，避免误挂一夜：

```bash
timeout 20m uv run python tests/verify_semantic_integrity.py --model ... --preset ...
```

**Lite vs HF（同一 tokenizer 字符串）**：可用 `--hf-same-as-lite` 或 `--hf-model` 指向 FP16 权重；仅对比 **prefill 最后一格 logits + 首个 greedy token** 时加 `--prefill-only`（更快）。与 Tier-B 对齐的 **chat 模板** 用 `--apply-chat-template auto`（对单条 user 文本包装）。示例：

```bash
uv run python tests/verify_semantic_integrity.py --model models/Qwen3.5-9B-FP16 --quant none \
  --hf-same-as-lite --hf-device cuda --prefill-only --apply-chat-template auto \
  --prompt "What is the capital of France?"
```

**35B MoE GGUF（无 HF 参考）**：用 `scripts/qwen35_moe_packed_lite_logits_audit.py dump` 保存 `logits_last`，终端会打印统计；离线查看：`uv run python scripts/qwen35_moe_packed_lite_logits_audit.py stats /path/to/dump.pt`。

**GGUF / MoE logits 分层对齐（推荐顺序）**

1. **有 HF 参考（优先 9B）**：`verify_semantic_integrity.py` 加 `--prefill-only`，同一 tokenizer 字符串下对比 **Lite prefill 最后一格 logits** 与 **HF**。**异目录**下 Lite **GGUF** 或 **AWQ** 与 FP16 参考对比时 CosSim 阈值略低于「同目录 FP16 vs FP16」（见 `PREFILL_COSIM_MIN_GGUF_VS_FP16` / `PREFILL_COSIM_MIN_AWQ_VS_FP16`）；**greedy argmax 仍须与 HF 一致** 才算通过。用于区分量化路径与参考权重的偏差。
2. **35B MoE GGUF（仅 Lite 内部）**：脚本 `qwen35_moe_packed_lite_logits_audit.py` 对 **同一 GGUF、同一 prompt** 分别 `FASTINFERENCE_QWEN35_MOE_PACKED_GGUF=1` 与 `=0` 各跑一次 `dump`（dense 侧常加 `--frugal`），再 `diff` 看 CosSim / MaxAbs / argmax；用于排除 **packed 布局与切片** 问题。**dense 加载峰值显存/内存更高**，若进程被系统杀死（常见 exit **137**），可换更大显存机器、或两台机器各跑一个 `dump` 后拷贝 `.pt` 再本地 `diff`。
3. **无整模第二路径时**：`weight-parity`（同脚本子命令）对单层单 expert 校验 **packed 行切片 dequant** 与 **整 tensor dequant** 是否一致，不依赖 Lite 引擎。

**结果如何映射到原因**

| 现象 | 更可能的问题 |
| --- | --- |
| 9B：`verify_semantic_integrity` 失败（CosSim/argmax） | loader、dequant、线性层实现；可并行跑 `scripts/qwen35_gguf_alignment_audit.py` 做张量级缩小范围 |
| 35B：`diff`（packed vs dense）差、`weight-parity` 失败 | packed MoE GGUF 布局或切片 dequant |
| 35B：`diff` 好，但与 HF 差（若能跑 HF） | 与 packed 无关，对比 HF 或量化误差 |
| 仅 `quality_bar_spotcheck` 差、上述 logits 对齐好 | chat 模板、采样、解码路径 |

**Qwen3.5（LiteEngine）**：默认 **整段 prompt 单次 prefill**（`max_model_len` 内），以保证主干 hidden 连续；显存紧张时可设 `FASTINFERENCE_LITE_PREFILL_CHUNK=512` 等缩小块长（长上下文时可能与 HF 不一致）。

`huggingface_hub snapshot_download` 若需停止：`pkill -9 -f snapshot_download`

## Layer 0 submodule alignment (`verify_layer0_submodule_alignment.py`)

用于对比 HF 与 Lite 在 **第 0 层** 各子模块激活（prefill）。更多说明见脚本顶部 docstring。

**结构 / 权重严格对齐（推荐）**：两侧使用**同一 FP16** 权重目录，Lite 关闭量化。

```bash
uv run python tests/verify_layer0_submodule_alignment.py \
  --model models/Qwen3.5-9B-FP16 --hf-model models/Qwen3.5-9B-FP16 \
  --quant none --include-embed
```

**AWQ Lite vs FP16 HF 参考**：量化线性层 CosSim 会偏低，属预期；可看 RMSNorm / embed 是否仍接近。

```bash
uv run python tests/verify_layer0_submodule_alignment.py \
  --model models/Qwen3.5-9B-AWQ --hf-model models/Qwen3.5-9B-FP16 \
  --quant awq
```

可选（更细诊断）：

- `--hf-device cuda`：HF 也在 GPU 上跑前向，减轻 **CPU FP32 RMSNorm vs CUDA FP16** 的系统误差。脚本会在 HF 前向期间**临时把 Lite 主干权重挪到 CPU**（KV cache 仍在 GPU），避免「双份整模」显存 OOM；HF 跑完再迁回 GPU。
- `--check-layer0-weights`：直接对比 layer0 各参数与 HF `state_dict` 是否一致（排除「权重没加载」）。
- `--compare-post-attn-residual`：对比进入 `post_attention_layernorm` 前的张量（即 **x + attn_out**），判断 MLP/post-norm 差是否主要由注意力残差引起。

查看内建示例：`uv run python tests/verify_layer0_submodule_alignment.py --help`

## Pytest

默认只自动收集 `test_*.py`、`*_test.py`。若要包含内核对比测试，请显式指定：

```bash
uv run pytest tests/test_*.py tests/lite_smoke_test.py tests/verify_kernel_parity.py -v
```
