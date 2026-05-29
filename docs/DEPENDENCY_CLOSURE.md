# Lite Engine 依赖闭包

从 `vllm/engine/lite_engine.py` 出发的 transitive import 分析。

## 必须保留（lite 路径 transitive import 覆盖）

- `vllm/engine/*` — 引擎控制平面（lite_engine, step_scheduler, runtime_factory, ...）
- `vllm/serving/config_builder.py` — 配置构建
- `vllm/adapters/*` — 模型适配器（base, gemma4, qwen3_5, llama, registry）
- `vllm/model_executor/models/gemma4.py` — Gemma4 模型定义
- `vllm/model_executor/models/qwen3_5.py` — Qwen3.5 模型定义
- `vllm/model_executor/models/llama.py` — Llama 模型定义
- `vllm/model_executor/layers/lite_linear.py` — Lite 线性层
- `vllm/model_executor/model_loader/*` — 模型加载
- `vllm/kernels/triton/*` — Triton 内核
- `vllm/config/*` — 基础配置 dataclass
- `vllm/entrypoints/*` — API 入口
- `vllm/sample/*` — 采样
- `vllm/inputs/*` — 输入处理
- `vllm/triton_utils/*` — Triton 工具
- `vllm/utils/*` — 通用工具函数
- `vllm/transformers_utils/*` — HF 工具
- `vllm/attention/*` — 注意力后端

## 安全可删（lite 路径无直接/间接依赖）

### 一级目录

| 目录 | 行数 | lite import | 非 lite 引用方 |
|------|------|-------------|---------------|
| `vllm/worker/` | 7,184 | 零 | `vllm/forward_context.py`, `vllm/executor/abstract.py`, `vllm/executor/uniproc_executor.py`, `vllm/attention/backends/utils.py`, `vllm/structured_output/utils.py`, `vllm/logging_utils/dump_input.py`, `vllm/model_executor/warmup/*` |
| `vllm/core/` | 2,871 | 零 | `vllm/v1_outputs.py`, `vllm/executor/abstract.py`, `vllm/executor/uniproc_executor.py`, `vllm/attention/backends/utils.py`, `vllm/structured_output/utils.py`, `vllm/logging_utils/dump_input.py` |
| `vllm/distributed/` | 46 | 零 | `vllm/model_executor/layers/mamba/*`, `vllm/model_executor/layers/fused_moe/*`, `vllm/model_executor/layers/vocab_parallel_embedding.py`, `vllm/model_executor/layers/kda.py`, `vllm/model_executor/layers/quantization/fp8.py`, `vllm/model_executor/model_loader/weight_utils.py`, `vllm/device_allocator/cumem.py` |
| `vllm/third_party/` | 5,127 | 零 | `vllm/utils/import_utils.py`（动态能力检测） |

### 引用方处理清单

- `vllm/forward_context.py:17` — `from vllm.worker.ubatch_utils import UBatchSlices`
- `vllm/executor/abstract.py:14,18` — `from vllm.core.sched.output import ...` + `from vllm.worker.worker_base import WorkerBase`
- `vllm/executor/uniproc_executor.py:7` — `from vllm.core.sched.output import ...`
- `vllm/attention/backends/utils.py:23-24` — `from vllm.core.sched.output import ...` + `from vllm.worker.gpu_input_batch import InputBatch`
- `vllm/structured_output/utils.py:20,29` — `from vllm.core.sched.output import ...` + `from vllm.worker.gpu_input_batch import InputBatch`
- `vllm/v1_outputs.py:13,16-17` — `from vllm.core.sched.output import ...` + `from vllm.distributed.kv_events import ...`
- `vllm/logging_utils/dump_input.py:12` — `from vllm.core.sched.output import SchedulerOutput`
- `vllm/device_allocator/cumem.py:33` — `from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary`
- `vllm/model_executor/layers/mamba/*` — `from vllm.distributed.*`（整个 mamba/ 自身是遗留层）
- `vllm/model_executor/layers/fused_moe/*` — `from vllm.distributed.*`, `from vllm.worker.ubatching import ...`
- `vllm/model_executor/layers/kda.py` — `from vllm.distributed import ...`
- `vllm/model_executor/layers/vocab_parallel_embedding.py` — `from vllm.distributed import ...`
- `vllm/model_executor/layers/quantization/fp8.py` — `from vllm.distributed import ...`
- `vllm/model_executor/model_loader/weight_utils.py:673` — lazy `from vllm.distributed import ...`
- `vllm/model_executor/warmup/kernel_warmup.py:15-16` — `from vllm.worker.gpu_model_runner import GPUModelRunner` + `from vllm.worker.gpu_worker import Worker`
- `vllm/model_executor/warmup/deep_gemm_warmup.py:8` — `from vllm.distributed.parallel_state import ...`
- `vllm/utils/import_utils.py` — 动态检测 `vllm.third_party.triton_kernels`

## 兼容性边界（lite 不依赖但外部 API 需要）

- `vllm/entrypoints/openai/*` — OpenAI API server（如有外部用户依赖）
- `vllm/entrypoints/cli/*` — CLI 入口
- `vllm/grpc/*` — gRPC（如有外部用户依赖）
