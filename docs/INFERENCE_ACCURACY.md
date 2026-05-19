# 推理准确度：验收标准、测试与代码改进

本文合并原 `QUALITY_BAR.md`、`INFERENCE_QUALITY_ASSESSMENT.md`、`INFERENCE_QUALITY_RUNBOOK.md`、`EXPECTED_ALIGNMENT_METRICS.md`、`QWEN35_9B_GGUF_ACCURACY_BASELINE.md`、`QWEN35_GGUF_LINEAR_LAYOUT.md`，作为 **Lite 推理质量验收** 的单一入口。

当前默认回归目标是 `TinyLlama-1.1B`、`Qwen3.5-9B-AWQ`、`Gemma4-26B-A4B-it-AWQ-4bit`、`Gemma4-31B-it-AWQ-4bit`。GGUF / 35B / DeepSeek 相关内容保留为实验或历史诊断 runbook，不属于当前默认支持承诺。

---

## 1. 三档验收目标（B / A-lite / A-strict）

| 档位 | 目标 | 典型手段 |
|------|------|----------|
| **B. 观感合理** | 续写通顺、不离题、少乱码；**不要求**与 HF logits 一致 | 人工抽检、固定 prompt、`tests/tools/quality_bar_spotcheck.py` 启发式（仅粗筛） |
| **A-lite. 关键点审计** | 在资源受限场景下验证主链路可用、无明显结构错误、text-only 路径可完成生成 | 轻量审计、固定 2 到 3 个 prompt、首 token / 结束条件 / 文本完整性检查（例如 `tests/tools/gemma4_single_prompt_smoke.py`） |
| **A-strict. 严格对齐** | 与 HF 在数值与 greedy token 上尽量一致 | Prefill logits **CosSim** / **MaxErr**、`tests/verify_semantic_integrity.py` |

默认策略：

- **参数量 <= 14B**：走 **A-strict + B**
- **参数量 > 14B**：走 **A-lite + B**
- 当前 Gemma4-26B-A4B 在 `tests/run_inference_correctness_regression.sh` 中作为例外：默认开启 **A-strict + A-lite + B**（strict 采用 sequential GPU reference）

产品可主要采用 **B**；内核、量化与加载器回归优先用 **A-strict**；大模型在本机资源不足时退化到 **A-lite**。

补充说明：

- 对 `Gemma4-31B-it-AWQ-4bit` 这类 31B 级模型，专项 `A-strict` 采用 **sequential GPU reference**，而不是 Lite 与 HF 同时驻留 GPU。
- 具体做法是：先运行 Lite prefill 并抓取 last-token logits；随后释放 LiteEngine 的 GPU 占用，再在同一张 GPU 上加载 Gemma4 reference，做一次 `prefill-only` 对拍。
- 因此，Gemma4 `A-strict` 的通过标准仍然是严格的 prefill 数值对齐，但执行方式是顺序式的，而不是双驻留并行式的。

**观感档**建议同时满足：可读性（少 ``）、连贯性（无灾难性重复）、与 prompt 大致相关；贪婪首 token 不离谱。**不能**用「降低标准」掩盖整段乱码或 CosSim 长期≈0 且不可读——属实现或权重问题。

**何时「降低标准」仍不够**：首 token 即异常、整段乱码；Lite 与 HF `generate` 反差极大；Prefill 与合理参考几乎正交且输出不可读。

---

## 2. Qwen3.5 GGUF：实验/历史诊断说明

本节记录 GGUF 加载和布局对齐的历史修复。当前默认支持和回归路径使用 `Qwen3.5-9B-AWQ`，不是 GGUF；以下内容用于维护实验路径或排查旧报告。

以下改动均在 `vllm/model_executor/model_loader/__init__.py`（及审计脚本）中，使 **llama.cpp 导出的 GGUF** 与 **Hugging Face `Qwen3_5ForCausalLM`** 的权重语义一致，从而让 **LiteEngine + GGUF** 的 prefill / greedy 与 HF 参考对齐（在相同 FP16/BF16 参考下）。

| 主题 | 问题 | 修复要点 |
|------|------|----------|
| **GatedDeltaNet V 头布局** | llama.cpp `_LinearAttentionVReorderBase` 将 HF 的 **grouped** V 头改为 GGML **tiled**；直接当 HF 用会导致线性注意力整段错位 | 对 `attn_qkv`（仅 V 段）、`attn_gate`、`ssm_out`、`ssm_alpha`、`ssm_beta` 及 `A_log`/`dt_bias`、卷积中 **V 通道段** 做**逆变换**：先 `reshape` 为 `(num_v_per_k, num_k_heads, head_dim)`，再交换与导出时相同的两维（**不是**误用对称的 `(num_k_heads, num_v_per_k, …)` reshape） |
| **深度卷积 `ssm_conv1d`** | 手写 `numpy` + `.t()` 与 GGUF 字节序不一致，得到错误形状（如 `[4,1,8192]`） | 对 F32 使用 `_dequantize_gguf_tensor(..., target_shape=[C,1,K])`，与 HF `safetensors` 的 contiguous 布局一致，再对 **V 段通道** 做上表逆重排 |
| **RMSNorm `+1` 偏移** | Qwen3Next/Qwen3.5 导出对多数 `*.norm.weight` 存 **γ+1**，**唯独** `linear_attn.norm`（`ssm_norm`）不加 | 加载后对一维 `norm.weight` **减 1**；**跳过** `blk.*.ssm_norm.weight` / `linear_attn.norm` |
| **二维权重 `[out,in]`** | GGUF 元数据常为 `[in,out]`，`gguf.dequantize` 结果需与 PyTorch `nn.Linear` 对齐 | `_dequantize_gguf_tensor`：在 `target_shape` 为二维时，若 `res.shape == (in,out)` 则 `return res.T.contiguous()`；避免对 embedding 等误 `view(target_shape)` |
| **全局 dtype** | `get_model` 曾一律 `float16`，与 `text_config.dtype`（如 **bfloat16**）不一致 | `_resolve_model_dtype_from_hf_config` + `model.to(dtype=...)`（safetensors 与 GGUF 路径均适用） |
| **审计与回归** | 脚本未随加载逻辑更新 | `tests/tools/qwen35_gguf_alignment_audit.py`：展开 `text_config`、线性张量逆重排、卷积 V 重排、双精度余弦 |

**修复后典型结果**（`Qwen3.5-9B-GGUF` Q4_K_M vs `Qwen3.5-9B-FP16`；默认 `FASTINFERENCE_KV_TYPE=turbo_int4`，严格对拍可按需改 `FASTINFERENCE_KV_TYPE=fp16|fp8`）：`verify_semantic_integrity --preset qwen35_9b_gguf` 的 Prefill **CosSim≈0.99**，greedy 首 token 与 HF 一致；`quality_bar_spotcheck` 在贪婪解码下可读性/连贯性可通过。

---

## 3. GGUF 张量映射与布局说明（实验路径）

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

**Qwen3.5 35B MoE（`Qwen3_5MoeForConditionalGeneration` / `model_type: qwen3_5_moe_text`）**：FFN 为 HF 同构的稀疏 MoE 块（`mlp.gate`、`mlp.experts.gate_up_proj` / `down_proj`、`mlp.shared_expert`、`mlp.shared_expert_gate`）。GGUF 中路由为 `ffn_gate_inp`；专家 `ffn_gate_exps` 与 `ffn_up_exps` 在加载时 `concat` 为 `gate_up_proj`；共享专家为 `ffn_*_shexp`；可选 `ffn_gate_sw_shexp` 对应 `shared_expert_gate`（若缺失则保留初始化并打日志）。线性注意力相关的 V 头逆重排与 RMSNorm `-1` 校正对 `qwen3_5_moe_text` 与稠密 `qwen3_5` 相同。验收：`tests/tools/qwen35_gguf_alignment_audit.py --moe`、`verify_semantic_integrity --preset qwen35_35b_moe_gguf`（需本地 GGUF + HF 参考；**显存**大，可先跑权重 CosSim 审计）。

HF `state_dict` 常用前缀 `model.language_model.layers.*`；审计脚本会尝试两种前缀。

### 3.2 二维权重与 `ssm_conv1d`

- 多数 GGUF 二维张量元数据为 `[in_dim, out_dim]`；与 `target_shape == (out, in)` 比较后决定转置。
- **`ssm_conv1d`**：F32 时对 `raw` 使用 **`view(*target_shape)`** 为 `[C,1,K]`，与 HF 一致（见第 2 节「深度卷积」）。

---

## 4. 自动化对齐预期（单元与端到端）

- **GatedDeltaNet chunk / paged prefill**：这些 GGUF/线性注意力专项测试曾用于历史对齐排查；当前默认回归不再包含对应独立入口。
- **一键打印**：`uv run python tests/tools/report_expected_alignment_metrics.py`。
- **端到端**：`verify_semantic_integrity` 的 CosSim 依赖权重、dtype、KV 精度；**以现场打印为准**。Qwen3.5 同目录 HF 参考若用 FLA，建议 **`--hf-device cuda`**。
- **`LiteConfig`**：`rope_parameters` 为 JSON `null` 时按空字典处理（见 `vllm/model_executor/models/lite_config.py`）。

---

## 5. 可执行命令（Runbook）

约定：仓库根目录，`PYTHONPATH=.`，路径换成本机 `models/...`。

### 5.1 B 档（观感）

```bash
export MODEL="${MODEL:-models/Qwen3.5-9B-AWQ}"
PYTHONPATH=. uv run python tests/tools/quality_bar_spotcheck.py \
  --model "$MODEL" --quant awq --prompt-subset minimal \
  --max-new-tokens 96 --temperature 0
```

- Prompt 集合：脚本内 `DEFAULT_SPOTCHECK_PROMPTS` / `MINIMAL_PROMPT_IDS`。
- 可选：`bash tests/tools/run_inference_quality_suite.sh`。
- **`temperature == 0`**：贪婪 argmax，与 A 档 greedy 一致；默认脚本可能为采样，需显式指定。
- 常用参数：`--prompt-subset minimal|full`、`--prompts-file`（覆盖子集）、`--json`、`--no-heuristics-fail`（启发式告警仍 **exit 0**）。启发式 **PASS/WARN 不等于语义合格**，需人工扫输出。
- **Gemma4-26B A4B hard gate（默认开启）**：当 `--model` 命中 `gemma-4-26b` 且 `a4b` 时，`quality_bar_spotcheck.py` 会启用模型专用严格规则，用于把「可跑通但不可读」输出判为失败。规则分两层：
  - 文本结构硬规则：`fragmented_short_run_salad`、`gemma4_26b_*token_soup*`、`gemma4_26b_mixed_token_garble` 等（拦截短词循环、混脚本胶水词、乱码碎片）。
  - 固定 prompt 语义锚点：仅对 `tests/tools/fixtures/gemma4_correctness_prompts_default.json` 里的 `gemma_*` prompt id 生效（例如 `gemma_en_capital` 需出现 `paris/巴黎`，`gemma_en_bst` 需出现 `tree + (node|left|right|binary)`）。
  - 注意：该 hard gate 是 **Gemma4-26B 定向门禁**，不是通用语义评分器；其目标是“当前乱码必失败”，而不是跨模型统一打分。
- **DeepSeek-V2-Lite-GGUF**：权重目录常无 `config.json`；`quality_bar_spotcheck` 会像 tokenizer 一样从同级 `DeepSeek-V2-Lite-Chat` 读取 `model_type` / MoE 字段，以便 Tier-B 默认采样与贪婪下的惩罚与 MoE GGUF 一致。加载器在 `rope_scaling.type=yarn` 时会把 `rope_type` 与 `factor` 同步进 `rope_parameters`，供 `DeepseekV2RotaryEmbedding` 走 YaRN。**MLA 的 `blk.*.attn_q` / `blk.*.attn_kv_b` 在反量化时必须向 `_dequantize_gguf_for_load` 传入 LiteLinear 的 `target_shape=(out,in)`**：否则 `gguf.dequantize` 已给出的正确布局会被「按 GGUF 元数据 reshape」成转置，导致 `q_proj`/`kv_b` 权重与 HF 几乎正交，推理不可读。可选数值稳定：`FASTINFERENCE_GGUF_DEQUANT_FP32=1`（反量化）、`FASTINFERENCE_DEEPSEEK_ATTN_FP32=1`（QK 注意力 FP32，见 `deepseek_v2`）。默认中间反量化对 `deepseek_v2` 为 float16；需要更低峰值内存时可设 `FASTINFERENCE_GGUF_DEQUANT_FP8=1`。Q4 GGUF 与 HF bf16 的 logits 仍可能明显漂移；B 档以观感与 `quality_bar_spotcheck` 为准，**不**以 HF logits 一致为完成条件。
- **`FASTINFERENCE_TIER_B_DEEPSEEK_GGUF_ONLY=1`**：强制在 **Q4 GGUF 权重** 上跑 B 档（不切换到同级 `DeepSeek-V2-Lite-Chat` bf16）。脚本会自动收紧 `top_p` / `top_k` 与重复类惩罚，减轻尾部噪声；**观感仍常明显差于 bf16**，不宜作为交付验收标准，仅适合 GGUF 路径冒烟或调参。产品级 B 档验收请**不要**设置该变量，沿用默认的 sibling bf16。

Gemma4-26B hard gate 复检命令（当前行为应对乱码返回非 0）：

```bash
FASTINFERENCE_KV_TYPE=turbo_int4 FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS=1 FASTINFERENCE_KV_MAX_MODEL_LEN=1024 \
PYTHONPATH=. uv run python tests/tools/quality_bar_spotcheck.py \
  --model models/gemma-4-26B-A4B-it-AWQ-4bit \
  --quant awq \
  --prompts-file tests/tools/fixtures/gemma4_correctness_prompts_default.json \
  --max-new-tokens 64 --temperature 0 --top-p 0.95 --top-k -1 \
  --gpu-memory-utilization 0.55 --max-model-len 1024
```

### 5.2 A-strict（Lite vs HF）

```bash
export MODEL="${MODEL:-models/Qwen3.5-9B-AWQ}"
export FASTINFERENCE_KV_TYPE="${FASTINFERENCE_KV_TYPE:-turbo_int4}"
PYTHONPATH=. uv run python tests/verify_semantic_integrity.py \
  --model "$MODEL" --preset qwen35_9b_awq --hf-model models/Qwen3.5-9B-FP16
```

**GGUF vs FP16 参考（实验路径）**：

```bash
PYTHONPATH=. uv run python tests/verify_semantic_integrity.py \
  --model models/Qwen3.5-9B-GGUF --preset qwen35_9b_gguf \
  --hf-model models/Qwen3.5-9B-FP16
```

**Qwen3.5 35B MoE GGUF（历史/实验路径，不属于默认支持面）**：

```bash
export FASTINFERENCE_KV_TYPE="${FASTINFERENCE_KV_TYPE:-turbo_int4}"
PYTHONPATH=. uv run python tests/tools/qwen35_gguf_alignment_audit.py \
  --gguf /path/to/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --hf-dir /path/to/Qwen3.5-35B-A3B --moe
PYTHONPATH=. uv run python tests/verify_semantic_integrity.py \
  --model /path/to/qwen35-moe.gguf --preset qwen35_35b_moe_gguf \
  --hf-model /path/to/Qwen3.5-35B-A3B --hf-device cuda
```

**逐层 / 最后一层 hidden**：

```bash
PYTHONPATH=. uv run python tests/tools/verify_qwen35_final_hidden_alignment.py \
  --model "$MODEL" --hf-model "$MODEL" --quant none --per-layer
```

**DeepSeek-V2-Lite（A 档，HF safetensors）**：用 **同一** `DeepSeek-V2-Lite-Chat` 目录作为 `--lite-model` 与 `--hf-model`，可对比 Lite 与 `transformers` 的 last-token logits / 逐层 hidden（`tests/tools/compare_hf_lite_deepseek_logits.py`、`tests/tools/compare_hf_lite_deepseek_layer_hiddens.py`）。`safetensors` 加载需对齐 HF 命名：`kv_a_proj_with_mqa` → Lite `kv_a_proj`、`mlp.gate` → `router`、各 `mlp.experts.{e}.*` 堆叠为 `experts_gate` / `experts_up` / `experts_down`（见 `vllm/model_executor/model_loader/__init__.py`）。**Q4 GGUF** 与 bf16 参考仍有量化差距；严格数值对齐宜先用 bf16 同权重验证实现，再评估 GGUF。

**第 0 层子模块**：历史排查曾使用独立 layer-0 对齐脚本；当前默认 runbook 以 `tests/verify_semantic_integrity.py` 和 Gemma4 专项审计为准。

### 5.3 A-lite（关键点审计，适用于 >14B）

```bash
PYTHONPATH=. uv run python tests/tools/gemma4_single_prompt_smoke.py \
  --model models/gemma-4-31B-it-AWQ-4bit \
  --prompt "Summarize what a binary search tree is in one short paragraph." \
  --max-new-tokens 32 --temperature 0 --max-model-len 512
```

- `A-lite` **不是**完整 HF 数值对拍。
- 目标是验证：模型能加载、text-only 主路径能结束生成、固定 prompt 集具备首 token、正常结束、输出非空且无明显退化。
- 对 `Gemma4-31B-it-AWQ-4bit` 这类 >14B 模型，`tests/run_inference_correctness_regression.sh` 默认执行 **A-lite + B**；`A-strict` 仍保留为专项、手动开启路径（见 `RUN_GEMMA4_A_TIER=1`）。
- Gemma4 专项 `A-strict` 入口为 `tests/tools/gemma4_prefill_strict_audit.py`。默认使用 `--hf-device cuda`，走 **sequential GPU reference**：先抓 Lite prefill logits，再释放 Lite GPU 占用，再加载 Gemma4 reference 到 GPU 做一次 `prefill-only` 对拍。
- Gemma4 默认 correctness prompt 集为 `tests/tools/fixtures/gemma4_correctness_prompts_default.json`；边界题与长尾退化调试 prompt 集为 `tests/tools/fixtures/gemma4_edge_prompts_debug.json`。
- Gemma4 `A-strict` 基线（CosSim + argmax）固定在 `tests/tools/fixtures/gemma4_a_strict_baseline.json`。可用 `RUN_GEMMA4_DIAGNOSTIC=1 uv run pytest tests/test_gemma4_diagnostics_warn_only.py -q` 做基线对比；该入口**仅告警不 gate**。
- Gemma4-26B `A-strict` 基线固定在 `tests/tools/fixtures/gemma4_26b_a_strict_baseline.json`。可用 `RUN_GEMMA4_26B_DIAGNOSTIC=1 uv run pytest tests/test_gemma4_26b_strict_warn_only.py -q` 或 `bash tests/run_gemma4_26b_diagnostics_warn_only.sh` 复检（**仅告警不 gate**）。

### 5.4 GGUF 权重审计（实验路径）

```bash
PYTHONPATH=. uv run python tests/tools/qwen35_gguf_alignment_audit.py \
  --gguf models/Qwen3.5-9B-GGUF/<name>.gguf \
  --hf-dir models/Qwen3.5-9B-FP16 --conv-check
```

**AWQ**（与 FP16 对比时注意 HF 是否完整加载 packed 权重）：`verify_layer0_submodule_alignment.py --quant awq`，见 `docs/usage/troubleshooting.md`。

### 5.5 单点漂移定位

| 主题 | 命令 |
|------|------|
| Conv+SiLU 中间量 | `tests/tools/qwen35_gated_delta_conv_alignment.py --hf-dir "$MODEL"` |
| Chunk vs FLA | `tests/tools/qwen35_chunk_gated_delta_alignment.py` |
| 引擎 | `FASTINFERENCE_LITE_PREFILL_CHUNK`、`vllm/engine/lite_engine.py` |

Gemma4 long decode 漂移诊断：

```bash
PYTHONPATH=. uv run python tests/tools/gemma4_layer_drift_diagnostic.py \
  --model models/gemma-4-31B-it-AWQ-4bit \
  --prompt-id short_hi \
  --checkpoints 16,24,32 \
  --max-new-tokens 48
```

- 默认使用 `tests/tools/fixtures/gemma4_edge_prompts_debug.json`
- 默认报告 `local` / `full` 两类层在目标 token 的 `cos_to_t1` 与 `cos_to_prev` 摘要
- 用于诊断 Gemma4 超短 prompt 下的长尾 greedy 发散，不作为默认 correctness gate
- Gemma4 drift 基线固定在 `tests/tools/fixtures/gemma4_layer_drift_baseline_short_hi.json`，通过 `tests/test_gemma4_diagnostics_warn_only.py` 比较（偏离时仅 warning）

Gemma4 已知现象（`short_hi`, greedy, `turbo_int4` KV）：

- 典型退化尾段为：正常问候后进入 `thought` / HTML 片段 / 重复问候混杂的长尾输出。
- 目前诊断结果更像 **超短 prompt 下的长尾 greedy 发散**，而不是单一结构层错误。
- 一次基线 trace（`max_new_tokens=32`, `16/24/32` token）中，`local` / `full` 两类层都能观察到漂移，但没有出现“只有 full 崩”或“只有 local 崩”的断点：
  - `token=16`
    - `local`: `cos_to_t1 mean=0.726229`, `min=0.139810`
    - `full`: `cos_to_t1 mean=0.719264`, `min=0.065373`
  - `token=24`
    - `local`: `cos_to_t1 mean=0.747082`, `cos_to_prev mean=0.817752`
    - `full`: `cos_to_t1 mean=0.777167`, `cos_to_prev mean=0.837632`
  - `token=32`
    - `local`: `cos_to_t1 mean=0.733025`, `cos_to_prev mean=0.814279`
    - `full`: `cos_to_t1 mean=0.747627`, `cos_to_prev mean=0.852596`
- 当前解读：`local` 层最小值更低，局部漂移略早，但整体上 `local/full` 是一起漂，不支持“full/sliding 混合结构存在单点实现 bug”的强结论。
- 因此默认 correctness 采用 Gemma 专用 prompt 集，不让 `short_hi` 这类边界题阻塞整条回归；`short_hi` 保留在 edge prompt 集里，专门用于长尾退化诊断。

---

## 6. 推荐抽检 Prompt（B 档）

建议温度 **0～0.7**，每模型至少 **3～5 条**。对 **Qwen Instruct** 等对话模型，`tests/tools/quality_bar_spotcheck.py` 默认使用 ``--chat-template auto``，将每条提示包装为 **user** 轮（与裸续写相比通常更连贯）。

### 英文常识

1. `What is the capital of France? Answer in a few words.`
2. `Explain what a binary search tree is in one short paragraph.`
3. `Write a Python function that returns the sum of a list of integers.`

### 中文

1. `法国的首都是哪里？请用一句话回答。`
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
| P3 | 工具与可见性 | `quality_bar_spotcheck.py`、本文档；**MoE packed vs 稠密 logits**：`tests/tools/qwen35_moe_packed_lite_logits_audit.py`（`dump`/`diff`/`weight-parity`） |

对比 GGUF 与 HF 前请确认 **同源 revision**；否则 CosSim 低可能来自权重差异而非实现 bug。

---

## 8. 维护说明

- 脚本与引擎变更时同步更新 **§4 预期** 与 **§5 命令**。
- 新增模型族时补充该族的典型合理行为与已知陷阱。
