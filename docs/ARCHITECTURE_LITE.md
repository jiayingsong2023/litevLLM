# litevLLM 架构解析

## 1. 核心设计：Operator 重定向 (Dispatching)

`litevLLM` 的核心魔法在于它对 `vLLM` 原有算子分发机制的拦截。

### 1.1 CustomOp 拦截 (`vllm/model_executor/custom_op.py`)
在原始 vLLM 中，算子优先尝试调用高性能的 C++ `_C` 绑定。在 `litevLLM` 中，我们通过以下修改强制走 Python 路径：
- 禁用了所有 C++ 后端的探测。
- 强制所有 GPU 平台返回 `self.forward_native`。
- `forward_native` 会调用 PyTorch 官方优化的算子（如 `torch.nn.functional.rms_norm`）或者我们自定义的 Triton Kernel。

### 1.2 动态 Mock 系统 (`vllm/_custom_ops.py`)
由于我们删除了 `csrc`，直接访问 `torch.ops._C` 会抛出 `AttributeError`。我们实现了一个 **MockNamespace** 系统：
- 当代码尝试访问缺失的 C++ 算子时，Mock 系统会捕获它。
- 如果是关键算子（如 MoE 路由），我们会重定向到我们手写的 **Native Fallback** 实现。
- 如果是非关键算子，它会抛出一个清晰的“简化版错误”，指导开发者补齐。

## 2. 算子补齐 (Kernel Fallbacks)

为了让 MoE 等复杂模型跑通，我们手动实现了以下关键算子的 Python/Native 版本：

- **`moe_align_block_size`**: 专家对齐逻辑，完全用 PyTorch 逻辑重写，解决了原有的递归死循环问题。
- **`topk_softmax` / `topk_sigmoid`**: MoE 路由的核心激活函数。
- **`moe_sum`**: 专家输出的加权求和聚合。
- **`weak_ref_tensor`**: 兼容 `torch.compile` 的轻量级张量生命周期管理。

## 3. 注意力机制后端 (`v1/attention/selector.py`)

在 `litevLLM` 中，我们通过硬编码禁用了对 `FlashAttention` 或 `Xformers`（通常需要 C++ 编译）的依赖，强制选择 **`TRITON_ATTN`**。
这意味着 PagedAttention 的逻辑完全由 Python 控制的 Triton Kernel (位于 `vllm/model_executor/layers/attention/backends/triton.py`) 完成。

## 4. 构建层简化

- **`setup.py`**: 被大幅删减，设置 `VLLM_TARGET_DEVICE="empty"`，移除了所有 `CMakeExtension`。这使得安装过程不需要任何 C++ 编译器和庞大的编译环境。
- **`CMakeLists.txt`**: 增加了一行 `return()`，彻底阻断 CMake 编译。

## 5. 当前限制与挑战

- **Eager 模式依赖**: 目前在使用 CUDA Graph 捕获时，Python 逻辑会触发流捕获不支持的错误。因此，目前推荐配合 `--enforce-eager` 使用。
- **量化缺失**: AWQ/GPTQ 的 C++ Kernel 已移除，目前只能运行 Unquantized (FP16/BF16) 模型。后续计划补齐 Triton 版本的量化算子。
