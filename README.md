# litevLLM (Triton/Python Only)

`litevLLM` 是一个基于 `vLLM` 深度简化的极致推理引擎。它完全移除了 C++/CUDA/ROCm 构建层，仅保留 Triton 和 Python 实现。

## 🌟 核心理念
- **极致简化**: 删除了所有的 `csrc` 目录（约数十万行 C++ 代码）。
- **Only Triton**: 强制使用 Triton 算子，完全不依赖自定义 C++ 扩展。
- **高可移植性**: 无需编译，只要有 PyTorch 和 Triton，即可在 AMD、NVIDIA 等多种 GPU 上运行。
- **架构透明**: 推理全流程由 Python 驱动，方便开发者进行二次开发和实验。

## 🚀 快速开始

### 环境要求
- **Python**: 3.12+
- **GPU**: AMD (ROCm 6.0+) 或 NVIDIA (CUDA 12.0+)
- **依赖**: PyTorch, Triton, `uv` (推荐)

### 安装步骤
```bash
# 克隆仓库
git clone https://github.com/jiayingsong2023/litevLLM.git
cd litevLLM

# 使用 uv 创建虚拟环境并安装 (无需编译)
uv pip install -e .
```

### 运行基准测试 (以 MoE 模型为例)
```bash
# 强制使用 Eager 模式运行 (目前推荐)
python -m vllm.entrypoints.cli.main bench latency \
    --model Qwen/Qwen1.5-MoE-A2.7B-Chat \
    --enforce-eager
```

## 🛠 当前支持
- **Attention**: 纯 Triton 版 PagedAttention.
- **MoE**: 纯 Python 调度的混合专家模型。
- **Platform**: 深度优化 AMD ROCm 7.1 兼容性。

## 📄 文档
更多细节请参考 [ARCHITECTURE_LITE.md](./docs/ARCHITECTURE_LITE.md)。
