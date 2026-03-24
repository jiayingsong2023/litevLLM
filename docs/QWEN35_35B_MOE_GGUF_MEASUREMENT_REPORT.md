# Qwen3.5 35B MoE GGUF 实测报告

## 1. 摘要

| 项目 | 结果 |
|------|------|
| **端到端推理（LiteEngine + GGUF）** | **未执行** — 本环境无 Qwen3.5 35B MoE GGUF 权重文件 |
| **与 HF 数值对拍（`verify_semantic_integrity`）** | **未执行** — 依赖同上 + 需同目录 HF 参考 |
| **权重 CosSim 审计（`qwen35_gguf_alignment_audit.py --moe`）** | **未执行** — 依赖 GGUF 与 HF 目录 |
| **代码与模块可导入** | **通过**（见 §3） |

**结论（当前环境）：** 无法在本地完成「35B MoE GGUF 真实推理」与 A 档对拍；需在具备 **GGUF 文件 + 足够显存（或 CPU 卸载方案）** 的机器上按 §4 复现。

---

## 2. 实测环境（自动探测）

| 项 | 值 |
|----|-----|
| 仓库路径 | `FastInference` 工作区 |
| PyTorch | `2.9.1+rocm7.2.0`（示例） |
| `torch.cuda.is_available()` | `True` |
| 设备名 | `Radeon 8060S Graphics`（集成 GPU / ROCm） |
| `nvidia-smi` | 不可用（ROCm 环境常见） |
| 仓库内 `*.gguf` | 未发现 `Qwen3.5*35B*` / `*Moe*` 35B 相关文件；存在其他模型 GGUF（如 `models/` 下非本任务权重） |

**资源说明：** 35B 级 MoE 即使量化后通常仍需 **数十 GB 级** 显存或内存；集成显卡一般 **不满足** 稳定加载与长上下文推理，实测应在 **工作站级 GPU** 上执行。

---

## 3. 已完成的轻量校验（不依赖 35B 权重）

以下用于确认实现可被解释器加载（不代表 35B 已跑通）：

```bash
cd /path/to/FastInference
PYTHONPATH=. uv run python -c "
from vllm.model_executor.models.lite_config import LiteConfig
from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeSparseMoeBlock
class C:
    model_type='qwen3_5_moe_text'
    hidden_size=64
    num_experts=8
    num_experts_per_tok=2
    moe_intermediate_size=16
    shared_expert_intermediate_size=16
    vocab_size=100
    num_hidden_layers=1
    linear_num_key_heads=2
    linear_num_value_heads=4
    linear_key_head_dim=8
    linear_value_head_dim=8
    linear_conv_kernel_dim=4
    num_attention_heads=4
    num_key_value_heads=2
    head_dim=16
    rope_parameters={}
lc = LiteConfig(C())
m = Qwen3_5MoeSparseMoeBlock(lc, None, 'mlp')
import torch
y = m(torch.randn(1, 2, 64))
assert y.shape == (1, 2, 64)
print('MoE block forward OK:', tuple(y.shape))
"
```

**预期：** 打印 `MoE block forward OK`；若失败则说明环境与依赖需先修复。

---

## 4. 在有权重机器上的复现实测（推荐）

### 4.1 B 档：观感 / 连贯性（启发式）

```bash
export FASTINFERENCE_KV_FP8=0
export MODEL=/path/to/Qwen3.5-35B-A3B-GGUF   # 含 config + tokenizer + .gguf
PYTHONPATH=. uv run python tests/tools/quality_bar_spotcheck.py \
  --model "$MODEL" --quant gguf \
  --prompt-subset minimal --max-new-tokens 96 --temperature 0
```

记录：脚本输出的启发式 PASS/WARN、生成文本片段（人工读连贯性）。

### 4.2 权重对齐（CosSim）

```bash
PYTHONPATH=. uv run python tests/tools/qwen35_gguf_alignment_audit.py \
  --gguf /path/to/model.gguf \
  --hf-dir /path/to/Qwen3.5-35B-A3B-HF \
  --moe
```

记录：各张量 CosSim / MaxErr、`[OK]` / `[CHECK]` 比例。

### 4.3 A 档：Prefill logits vs HF

```bash
export FASTINFERENCE_KV_FP8=0
PYTHONPATH=. uv run python tests/verify_semantic_integrity.py \
  --model /path/to/gguf_or_dir \
  --preset qwen35_35b_moe_gguf \
  --hf-model /path/to/Qwen3.5-35B-A3B-HF \
  --hf-device cuda
```

记录：终端打印的 CosSim、MaxErr、greedy 首 token 是否一致。

---

## 5. 实测报告模板（供你填写）

在目标机器跑完 §4 后，可将下表补全为正式结论：

| 检查项 | 命令/预设 | 结果 | 备注 |
|--------|-----------|------|------|
| MoE 审计 | `qwen35_gguf_alignment_audit.py --moe` |  |  |
| 语义对拍 | `verify_semantic_integrity --preset qwen35_35b_moe_gguf` |  |  |
| 观感抽检 | `quality_bar_spotcheck.py --quant gguf` |  |  |
| 硬件 | GPU 型号 / 显存 |  |  |
| GGUF 量化档 | 如 Q4_K_M |  |  |

---

## 6. 参考文档

- 合并说明与映射：`docs/INFERENCE_ACCURACY.md`（含 35B MoE 段落）
- 实现要点：MoE FFN 与 GGUF 映射见 `vllm/model_executor/models/qwen3_5.py`、`vllm/model_executor/model_loader/__init__.py`
