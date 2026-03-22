# 推理准确度：验收标准、测试与代码改进

本文合并原 `QUALITY_BAR.md`、`INFERENCE_QUALITY_ASSESSMENT.md`、`INFERENCE_QUALITY_RUNBOOK.md`、`EXPECTED_ALIGNMENT_METRICS.md`、`QWEN35_9B_GGUF_ACCURACY_BASELINE.md`、`QWEN35_GGUF_LINEAR_LAYOUT.md`，作为 **Lite 推理质量与 Qwen3.5 GGUF 加载** 的单一入口。

---

## 1. 两档验收目标（B / A）

| 档位 | 目标 | 典型手段 |
|------|------|----------|
| **B. 观感合理** | 续写通顺、不离题、少乱码；**不要求**与 HF logits 一致 | 人工抽检、固定 prompt、`scripts/quality_bar_spotcheck.py` 启发式（仅粗筛） |
| **A. 严格对齐** | 与 HF 在数值与 greedy token 上尽量一致 | Prefill logits **CosSim** / **MaxErr**、`tests/verify_semantic_integrity.py` |

产品可主要采用 **B**；内核、量化与加载器回归用 **A**。

**观感档**建议同时满足：可读性（少 ``）、连贯性（无灾难性重复）、与 prompt 大致相关；贪婪首 token 不离谱。**不能**用「降低标准」掩盖整段乱码或 CosSim 长期≈0 且不可读——属实现或权重问题。

**何时「降低标准」仍不够**：首 token 即异常、整段乱码；Lite 与 HF `generate` 反差极大；Prefill 与合理参考几乎正交且输出不可读。

---

## 2. Qwen3.5 GGUF：推理准确度相关代码改进（总结）

以下改动均在 `vllm/model_executor/model_loader/__init__.py`（及审计脚本）中，使 **llama.cpp 导出的 GGUF** 与 **Hugging Face `Qwen3_5ForCausalLM`** 的权重语义一致，从而让 **LiteEngine + GGUF** 的 prefill / greedy 与 HF 参考对齐（在相同 FP16/BF16 参考下）。

| 主题 | 问题 | 修复要点 |
|------|------|----------|
| **GatedDeltaNet V 头布局** | llama.cpp `_LinearAttentionVReorderBase` 将 HF 的 **grouped** V 头改为 GGML **tiled**；直接当 HF 用会导致线性注意力整段错位 | 对 `attn_qkv`（仅 V 段）、`attn_gate`、`ssm_out`、`ssm_alpha`、`ssm_beta` 及 `A_log`/`dt_bias`、卷积中 **V 通道段** 做**逆变换**：先 `reshape` 为 `(num_v_per_k, num_k_heads, head_dim)`，再交换与导出时相同的两维（**不是**误用对称的 `(num_k_heads, num_v_per_k, …)` reshape） |
| **深度卷积 `ssm_conv1d`** | 手写 `numpy` + `.t()` 与 GGUF 字节序不一致，得到错误形状（如 `[4,1,8192]`） | 对 F32 使用 `_dequantize_gguf_tensor(..., target_shape=[C,1,K])`，与 HF `safetensors` 的 contiguous 布局一致，再对 **V 段通道** 做上表逆重排 |
| **RMSNorm `+1` 偏移** | Qwen3Next/Qwen3.5 导出对多数 `*.norm.weight` 存 **γ+1**，**唯独** `linear_attn.norm`（`ssm_norm`）不加 | 加载后对一维 `norm.weight` **减 1**；**跳过** `blk.*.ssm_norm.weight` / `linear_attn.norm` |
| **二维权重 `[out,in]`** | GGUF 元数据常为 `[in,out]`，`gguf.dequantize` 结果需与 PyTorch `nn.Linear` 对齐 | `_dequantize_gguf_tensor`：在 `target_shape` 为二维时，若 `res.shape == (in,out)` 则 `return res.T.contiguous()`；避免对 embedding 等误 `view(target_shape)` |
| **全局 dtype** | `get_model` 曾一律 `float16`，与 `text_config.dtype`（如 **bfloat16**）不一致 | `_resolve_model_dtype_from_hf_config` + `model.to(dtype=...)`（safetensors 与 GGUF 路径均适用） |
| **审计与回归** | 脚本未随加载逻辑更新 | `scripts/qwen35_gguf_alignment_audit.py`：展开 `text_config`、线性张量逆重排、卷积 V 重排、双精度余弦 |

**修复后典型结果**（`Qwen3.5-9B-GGUF` Q4_K_M vs `Qwen3.5-9B-FP16`，`FASTINFERENCE_KV_FP8=0`）：`verify_semantic_integrity --preset qwen35_9b_gguf` 的 Prefill **CosSim≈0.99**，greedy 首 token 与 HF 一致；`quality_bar_spotcheck` 在贪婪解码下可读性/连贯性可通过。

---

## 3. GGUF 张量映射与布局说明

### 3.1 命名映射（`map_rules` 摘要）

| GGUF 名 | Lite 键（`model.layers.{i}.` 下） | PyTorch `weight` 形状 `[out, in]`（9B 示例） |
|---------|-----------------------------------|---------------------------------------------|
| `token_embd.weight` | `embed_tokens` | `[vocab, hidden]` |
| `blk.{i}.attn_qkv.weight` | `linear_attn.in_proj_qkv` | `[8192, 4096]` |
| `blk.{i}.attn_gate.weight` | `linear_attn.in_proj_z` | `[4096, 4096]` |
| `blk.{i}.ssm_out.weight` | `linear_attn.out_proj` | `[4096, 4096]` |
| `blk.{i}.ssm_alpha/beta.weight` | `in_proj_a` / `in_proj_b` | `[32, 4096]` |
| `blk.{i}.ssm_conv1d.weight` | `linear_attn.conv1d` | `[conv_dim, 1, kernel]` |
| `blk.{i}.ffn_*.weight` | `mlp.*` | 见 MLP 标准形状 |

HF `state_dict` 常用前缀 `model.language_model.layers.*`；审计脚本会尝试两种前缀。

### 3.2 二维权重与 `ssm_conv1d`

- 多数 GGUF 二维张量元数据为 `[in_dim, out_dim]`；与 `target_shape == (out, in)` 比较后决定转置。
- **`ssm_conv1d`**：F32 时对 `raw` 使用 **`view(*target_shape)`** 为 `[C,1,K]`，与 HF 一致（见第 2 节「深度卷积」）。

---

## 4. 自动化对齐预期（单元与端到端）

- **GatedDeltaNet chunk**：Lite `_torch_chunk_gated_delta_rule` vs FLA naive — 单元测试 `tests/test_qwen35_chunk_gated_delta_rule.py`（需安装 `flash-linear-attention`，否则跳过）。
- **Paged prefill**：`tests/test_qwen35_paged_prefill_vs_torch_reference.py`（`FASTINFERENCE_KV_FP8=0`）。
- **一键打印**：`uv run python scripts/report_expected_alignment_metrics.py`。
- **端到端**：`verify_semantic_integrity` 的 CosSim 依赖权重、dtype、KV 精度；**以现场打印为准**。Qwen3.5 同目录 HF 参考若用 FLA，建议 **`--hf-device cuda`**。
- **`LiteConfig`**：`rope_parameters` 为 JSON `null` 时按空字典处理（见 `tests/test_lite_config_rope_parameters.py`）。

---

## 5. 可执行命令（Runbook）

约定：仓库根目录，`PYTHONPATH=.`，路径换成本机 `models/...`。

### 5.1 B 档（观感）

```bash
export MODEL="${MODEL:-models/Qwen3.5-9B-FP16}"
PYTHONPATH=. uv run python scripts/quality_bar_spotcheck.py \
  --model "$MODEL" --quant none --prompt-subset minimal \
  --max-new-tokens 96 --temperature 0
```

- Prompt 集合：脚本内 `DEFAULT_SPOTCHECK_PROMPTS` / `MINIMAL_PROMPT_IDS`。
- 可选：`bash scripts/run_inference_quality_suite.sh`。
- **`temperature == 0`**：贪婪 argmax，与 A 档 greedy 一致；默认脚本可能为采样，需显式指定。
- 常用参数：`--prompt-subset minimal|full`、`--prompts-file`（覆盖子集）、`--json`、`--no-heuristics-fail`（启发式告警仍 **exit 0**）。启发式 **PASS/WARN 不等于语义合格**，需人工扫输出。

### 5.2 A 档（Lite vs HF）

```bash
export MODEL="${MODEL:-models/Qwen3.5-9B-FP16}"
export FASTINFERENCE_KV_FP8="${FASTINFERENCE_KV_FP8:-0}"
PYTHONPATH=. uv run python tests/verify_semantic_integrity.py \
  --model "$MODEL" --preset qwen35_9b_fp16 --hf-model "$MODEL" --hf-device cuda
```

**GGUF vs FP16 参考**：

```bash
PYTHONPATH=. uv run python tests/verify_semantic_integrity.py \
  --model models/Qwen3.5-9B-GGUF --preset qwen35_9b_gguf \
  --hf-model models/Qwen3.5-9B-FP16
```

**逐层 / 最后一层 hidden**：

```bash
PYTHONPATH=. uv run python scripts/verify_qwen35_final_hidden_alignment.py \
  --model "$MODEL" --hf-model "$MODEL" --quant none --per-layer
```

**第 0 层子模块**：

```bash
PYTHONPATH=. uv run python tests/verify_layer0_submodule_alignment.py \
  --model "$MODEL" --hf-model "$MODEL" --quant none --include-embed
```

### 5.3 GGUF 权重审计

```bash
PYTHONPATH=. uv run python scripts/qwen35_gguf_alignment_audit.py \
  --gguf models/Qwen3.5-9B-GGUF/<name>.gguf \
  --hf-dir models/Qwen3.5-9B-FP16 --conv-check
```

**AWQ**（与 FP16 对比时注意 HF 是否完整加载 packed 权重）：`verify_layer0_submodule_alignment.py --quant awq`，见 `docs/usage/troubleshooting.md`。

### 5.4 单点漂移定位

| 主题 | 命令 |
|------|------|
| Conv+SiLU 中间量 | `scripts/qwen35_gated_delta_conv_alignment.py --hf-dir "$MODEL"` |
| Chunk vs FLA | `scripts/qwen35_chunk_gated_delta_alignment.py` |
| 引擎 | `FASTINFERENCE_LITE_PREFILL_CHUNK`、`vllm/engine/lite_engine.py` |

---

## 6. 推荐抽检 Prompt（B 档）

建议温度 **0～0.7**，每模型至少 **3～5 条**。

### 英文常识

1. `The capital of France is`
2. `Explain what a binary search tree is in one short paragraph.`
3. `Write a Python function that returns the sum of a list of integers.`

### 中文

1. `法国的首都是`
2. `用两三句话解释什么是梯度下降。`
3. `请用 Python 写一个简单的 Hello World 程序。`

### 代码 / 格式

1. `def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n    `
2. `{"name": "test", `

### 短句边界

1. `Hi,`

---

## 7. 排查优先级（简表）

| 优先级 | 假设 | 路径 |
|--------|------|------|
| P0 | 主干前向与 HF 不一致（与量化无关） | 先收紧 FP16/BF16 同目录 A 档、RoPE、解码循环 |
| P1 | GGUF 映射 / 反量化 / 布局 | `model_loader/__init__.py`、§2、§3、`qwen35_gguf_alignment_audit.py` |
| P2 | AWQ pack / matmul | `tensor.py`、`qwen3_5.py` |
| P3 | 工具与可见性 | spotcheck、本文档 |

对比 GGUF 与 HF 前请确认 **同源 revision**；否则 CosSim 低可能来自权重差异而非实现 bug。

---

## 8. 维护说明

- 脚本与引擎变更时同步更新 **§4 预期** 与 **§5 命令**。
- 新增模型族时补充该族的典型合理行为与已知陷阱。
