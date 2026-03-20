# Tests

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
