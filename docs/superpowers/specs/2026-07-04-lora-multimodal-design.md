# LoRA + Multimodal 支持设计

**Date:** 2026-07-04  
**Author:** Kimi Code CLI  
**Status:** Design ready for review

## 1. 背景与目标

当前项目已具备 LiteEngine 控制面、标准模型执行路径（Llama/Qwen/Gemma4）和 DeepSeek 自定义路径。但服务能力仍有两块明显缺口：

1. **LoRA**：入口参数已经存在（`lora_id` / `lora_request`），`LoRARuntimeRegistry` 和 `InputBatchBuilder` 已经能构造 `lora_mapping`，但 model 层没有真正加载/应用 LoRA weight。
2. **Multimodal**：`LiteMultiModalProcessor` 只能处理单请求图片 batch，Qwen2VL 只有 placeholder vision projector，Gemma4 直接拒绝 multimodal 输入。

### 1.1 设计目标

- 让 LiteEngine 标准模型支持 **LoRA adapter 推理**（Phase 1：单个 batch 内只允许一个 active LoRA；Phase 3：mixed LoRA batch）。
- 让 Gemma4 先支持 **单请求单图输入**（Phase 2A：真实 placeholder 替换、vision tower、可回归）。
- 再扩展到 **多图、多请求 continuous batching、Qwen2VL**（Phase 2B）。
- 不改动 DeepSeek 自定义路径（DeepSeek adapter 继续拒绝 multimodal / LoRA）。

### 1.2 非目标

- 不支持 DeepSeek 的 LoRA / multimodal。
- 不支持训练 LoRA，只支持加载已有 PEFT checkpoint。
- 第一阶段不支持 mixed LoRA batch 和 multimodal LoRA。
- 不支持 audio / video（预留接口但不实现）。

---

## 2. LoRA 设计

### 2.1 当前状态

已有代码：

- `vllm/lora/request.py`: `LoRARequest` dataclass。
- `vllm/engine/lora_runtime.py`: `LoRARuntimeRegistry`，负责注册/解析 adapter。
- `vllm/engine/lite_engine.py`: `add_request()` 调用 `lora_registry.resolve_adapter()`，并把 `lora_id` / `lora_int_id` / `lora_path` 写入 `RequestState`。
- `vllm/engine/input_batch_builder.py`: 构造 `attn_metadata["lora_mapping"]`、`lora_adapter_count`、`mixed_lora_batch`。
- `vllm/model_executor/models/llama.py` / `gemma4/` / `qwen3_5.py`: forward 已经传递 `lora_mapping` 到 `LiteLinear`。
- `vllm/model_executor/models/_fused_awq_pair.py`: LoRA active 时自动禁用 AWQ fusion。

缺失：

- `LiteLinear` 不消费 `lora_mapping`。
- 没有加载 LoRA weight 的逻辑。
- 没有把 LoRA weight 绑定到模型层的机制。
- `ModelCapabilities` 里没有 `supports_lora` 标志，engine 无法早期拒绝不支持 LoRA 的请求。

### 2.2 LoRA weight 表示（runtime 形状）

每个可 LoRA 化层维护一个 `LoRALayerWeights`。运行时计算为 `delta = x @ lora_a @ lora_b * scaling`：

```python
class LoRALayerWeights:
    def __init__(
        self,
        *,
        lora_name: str,
        rank: int,
        alpha: int,
        lora_a: torch.Tensor,   # (input_size, rank)
        lora_b: torch.Tensor,   # (rank, output_size)
    ) -> None:
        self.lora_name = lora_name
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_a = lora_a
        self.lora_b = lora_b
```

支持 LoRA 的目标层（Phase 1）：

- `q_proj`, `k_proj`, `v_proj`, `o_proj`
- `gate_proj`, `up_proj`, `down_proj`

Phase 1 不支持的：

- `lm_head` LoRA；
- `adapter_model.bin` 格式，只支持 `adapter_model.safetensors`；
- mixed LoRA batch。

### 2.3 加载 LoRA checkpoint

新增 `vllm/lora/loader.py`：

```python
class LoRALoader:
    def __init__(self, base_model: nn.Module) -> None: ...

    def load_adapter(
        self,
        *,
        lora_name: str,
        lora_path: str,
    ) -> dict[str, LoRALayerWeights]:
        """
        从 Safetensors PEFT checkpoint 加载 LoRA weight。
        返回：layer full name -> LoRALayerWeights
        """
        ...
```

实现要点：

- 只读取 `adapter_model.safetensors`（Phase 1 不支持 .bin）。
- 读取 `adapter_config.json` 拿 `r`, `lora_alpha`, `target_modules`。
- **PEFT 常见保存形状**：`lora_A.weight` 是 `(rank, input_size)`，`lora_B.weight` 是 `(output_size, rank)`。
- **Loader 必须在加载时转置**：把 A 转成 `(input_size, rank)`，B 转成 `(rank, output_size)`。
- 如果转置后 shape 和对应 `LiteLinear` 的 `(input_size, output_size)` 对不上，立即 raise `ValueError`。
- Safetensors key 映射到 base model 的 `LiteLinear` prefix（例如 `base_model.model.layers.0.self_attn.q_proj` -> `model.layers.0.self_attn.q_proj`）。
- 如果 `target_modules` 是列表（如 `["q_proj", "v_proj"]`），只加载这些层。

### 2.4 绑定 LoRA weight 到模型层

方案：给 `LiteLinear` 增加一个 `LoRAManager` 引用，而不是让每个 layer 直接持有所有 adapter 的 weight。

```python
class LoRAManager:
    def bind_to_model(self, base_model: nn.Module) -> None:
        """遍历 base_model 中所有 LiteLinear，注入 self 和 target_name。"""
        for name, module in base_model.named_modules():
            if isinstance(module, LiteLinear):
                module.lora_manager = self
                module.lora_target_name = name


class LiteLinear(nn.Module):
    def __init__(...):
        ...
        self.lora_manager: LoRAManager | None = None
        self.lora_target_name: str | None = None  # 例如 "model.layers.0.self_attn.q_proj"

    def forward(self, x: torch.Tensor, lora_mapping: Any = None, **kwargs) -> torch.Tensor:
        base_out = self._base_forward(x, **kwargs)
        if self.lora_manager is None or lora_mapping is None:
            return base_out
        lora_delta = self.lora_manager.compute_delta(
            target_name=self.lora_target_name,
            x=x,
            lora_mapping=lora_mapping,
        )
        if lora_delta is None:
            return base_out
        return base_out + lora_delta
```

`LiteEngine` 在初始化 model 后调用 `lora_manager.bind_to_model(model)`。

### 2.5 单 active LoRA per batch（Phase 1）

Phase 1 约束：**一个 batch 内所有请求要么都没 LoRA，要么都用同一个 LoRA**。这样 `LoRAManager.compute_delta()` 可以只查一个 adapter。

调度侧保证：

- `StepScheduler` 在选择 prefill/decode batch 时，按 `lora_id` 分组；同一 batch 内 `lora_id` 必须相同。
- `InputBatchBuilder` 只做防御性 assert：如果发现 mixed LoRA batch，直接报错（不应该走到这里）。

`LoRAManager` Phase 1 接口：

```python
class LoRAManager:
    def __init__(self, base_model: nn.Module) -> None:
        self._adapters: dict[str, dict[str, LoRALayerWeights]] = {}

    def register_adapter(
        self,
        *,
        lora_name: str,
        lora_path: str,
    ) -> None:
        """从 Safetensors PEFT checkpoint 加载并缓存一个 adapter 的所有 layer weights。"""
        ...

    def compute_delta(
        self,
        target_name: str,
        x: torch.Tensor,
        lora_mapping: list[str | None],
    ) -> torch.Tensor | None:
        """
        如果 batch 内没有 active LoRA，返回 None。
        如果有 active LoRA，取出该 adapter 对应 target_name 的 weights 计算 x @ A @ B * scaling。
        """
        active = {name for name in lora_mapping if name is not None}
        if not active:
            return None
        assert len(active) == 1, "Phase 1 does not support mixed LoRA"
        adapter_name = active.pop()
        weights = self._adapters[adapter_name].get(target_name)
        if weights is None:
            return None
        x2 = x.reshape(-1, x.shape[-1])
        h = x2 @ weights.lora_a  # (M, rank)
        delta = h @ weights.lora_b * weights.scaling  # (M, output_size)
        return delta.reshape(*x.shape[:-1], -1)
```

### 2.6 Mixed LoRA batch（Phase 2）

后续支持：

- `LoRAManager` 维护 stacked `lora_a` / `lora_b` tensors（所有 active adapters 拼成一个大张量）。
- `lora_mapping` 升级为 `LoRAMapping` dataclass，包含 `index_mapping: list[int]`（per-token / per-request adapter index）。
- `compute_delta()` 根据 index 做 `index_select` 拿到对应 A/B，再逐 token 计算。
- 这是 vLLM 的标准做法，但 Phase 1 先不做。

### 2.7 与量化/AWQ 的交互

- LoRA delta 用 float（或 base dtype）计算。
- base output 仍走量化 kernel。
- AWQ fusion 在 LoRA active 时自动禁用（已有 `_fused_awq_pair.py` 逻辑）。
- 注意：如果 base weight 是 AWQ int4，计算出的 delta 要和量化输出同 dtype 相加。

### 2.8 入口/注册

`LiteEngine` 已提供：

```python
def register_lora_adapter(self, lora_name: str, lora_path: str) -> LoRARequest:
    req = self.lora_registry.register_adapter(lora_name=lora_name, lora_path=lora_path)
    self.lora_manager.register_adapter(lora_name=lora_name, lora_path=lora_path)
    return req
```

需要补 `self.lora_manager` 初始化和调用。

---

## 3. Multimodal 设计

### 3.1 当前状态

已有代码：

- `vllm/engine/multimodal_processor.py`: `LiteMultiModalProcessor` / `NullMultiModalProcessor`。
- `vllm/engine/request_state.py`: `multi_modal_data`, `multi_modal_inputs`, `is_multimodal`。
- `vllm/engine/request_builder.py`: 把 `multi_modal_data` 写入 `RequestState`。
- `vllm/model_executor/models/qwen2_vl.py`: 有 `vision_projector` 和 `get_multimodal_embeddings` placeholder。
- `vllm/model_executor/models/gemma4/model.py`: `_assert_text_only_kwargs()` 直接拒绝 multimodal。
- `vllm/transformers_utils/configs/gemma4.py`: `Gemma4Config` 包含 `vision_config`。

缺失：

- `LiteMultiModalProcessor` 只支持单请求 image batch。
- 没有真正的 vision encoder（Qwen2VL 和 Gemma4 都只有 projector/placeholder）。
- 没有 image token 占位符替换、位置编码、slot mapping 处理。
- `InputBatchBuilder` 没有把 multimodal embeddings 传入 model forward。

### 3.2 高层流程

```text
User prompt with image(s)
       │
       ▼
  LiteMultiModalProcessor.prepare_before_tokenize()
       - 加载图片 -> resize/normalize -> prepared_image
       - 根据图片尺寸计算 image_token_count
       - 返回 (expanded_prompt, image_token_count, prepared_image)
       │
       ▼
  LiteRequestBuilder 用 expanded_prompt 调用 tokenizer.encode()
       - input_ids 已包含 N 个 image placeholder token
       - 构造 RequestState，multi_modal_data 包含 prepared_image
       │
       ▼
  LiteMultiModalProcessor.prepare_request()
       - prepared_image -> pixel_values
       - 写入 RequestState.multi_modal_inputs
       │
       ▼
  LiteMultiModalProcessor.build_prefill_inputs()
       - 聚合 batch 内所有请求的 pixel_values
       │
       ▼
  PrefillExecutor.execute()
       - 调用 model.get_multimodal_embeddings(pixel_values) -> image_embeddings
       - 把 image_embeddings 注入到 input_embeddings 的 placeholder 位置
       │
       ▼
  正常 prefill forward
```

**关键前提**：tokenizer / chat template 必须在 prompt 中把图片位置编码成 placeholder token id。如果 chat template 返回的是字符串 `<image>`，则 `prepare_before_tokenize()` 负责把它替换成 `image_token_id` 重复 N 次。

### 3.3 Image 预处理

扩展 `LiteMultiModalProcessor`：

```python
class LiteMultiModalProcessor:
    def __init__(self, *, model: Any, device: torch.device):
        ...
        self.image_config = self._build_image_config(model)

    def _build_image_config(self, model: Any) -> ImageConfig:
        """从 hf_config 读取 image_size, patch_size, num_channels 等。"""
        ...

    def _load_image(self, url: str) -> Image.Image: ...

    def _image_to_pixel_values(self, image: Image.Image) -> torch.Tensor:
        """resize -> to_tensor -> normalize -> add batch dim"""
        ...
```

`ImageConfig`：

```python
@dataclass
class ImageConfig:
    image_size: int
    patch_size: int
    num_channels: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    image_token_id: int | None = None
```

不同模型读取方式不同：

- **Gemma4**: 从 `hf_config.vision_config` 读 `image_size`, `patch_size`，placeholder token id 从 tokenizer 拿。

### 3.4 Image token 计数与 placeholder 展开

**核心约定**：`input_ids` 在 tokenizer encode 之前必须把图片位置展开成 **N 个 image placeholder tokens**。`InputBatchBuilder` 不再额外加 `image_token_count`。

当前 `LiteRequestBuilder.build()` 先 `tokenizer.encode(prompt)` 再构造 `RequestState`，这和展开 placeholder 的顺序冲突。需要新增一个 **pre-tokenize multimodal preparation** 步骤：

```python
class LiteMultiModalProcessor:
    def prepare_before_tokenize(
        self,
        prompt: str,
        multi_modal_data: dict[str, Any],
    ) -> tuple[str, int, Any]:
        """
        在 tokenizer encode 前调用。
        返回：(expanded_prompt, image_token_count, prepared_image)
        """
        ...
```

`LiteRequestBuilder.build()` 新流程：

1. 如果 `multi_modal_data` 非空，调用 `self.multimodal_processor.prepare_before_tokenize(prompt, multi_modal_data)`，得到 `expanded_prompt`、`image_token_count`、`prepared_image`。
2. `input_ids = self.tokenizer.encode(expanded_prompt)`。
3. 构造 `RequestState`，其中 `multi_modal_data` 包含 `prepared_image`，`image_token_count` 写入 `multi_modal_inputs`。
4. `LiteMultiModalProcessor.prepare_request(request_state)` 在 request state 构造后调用，只负责把 `prepared_image` 转成 `pixel_values`。

如果 chat template 本身已经展开 N 个 placeholder，则 `prepare_before_tokenize` 只需要验证/转义，不需要二次展开。

对固定 grid 的 vision encoder：

```python
def num_image_tokens(image_size: int, patch_size: int) -> int:
    grid = image_size // patch_size
    return grid * grid
```

Phase 2A 只支持固定 grid（Gemma4）。动态分辨率（Qwen2VL）放到 Phase 2B。

### 3.5 模型 forward 改造（Phase 2A 仅 Gemma4）

Gemma4 是多模态 decoder-only 模型，vision encoder 输出的 embeddings 通过 connector 接到语言模型：

- 新增 `Gemma4VisionTower` 和 `Gemma4Connector`（或从 HF checkpoint 加载）。
- `Gemma4ForConditionalGeneration.forward()` 接收 `pixel_values`：
  1. text embedding：`self.model.embed_tokens(input_ids)`。
  2. image embedding：`vision_tower(pixel_values) -> connector -> (num_image_tokens, hidden_size)`。
  3. 在 placeholder token 位置替换 image embedding。
  4. 删除 `_assert_text_only_kwargs()` 中的 `pixel_values` / `image_embeds` 限制。

### 3.6 InputBatchBuilder 改造

当前 `build_prefill_batch`：

- 对 prefill 请求，假设 `curr_input` 长度就是 `len(input_ids)`。
- multimodal 请求中，`input_ids` 已经由 request builder 展开了 N 个 image placeholder token，所以 `len(input_ids)` 就是真实 seq_len。

需要：

1. `RequestState` 增加 `image_token_count: int`（从 `multi_modal_inputs` 读取）。
2. `build_prefill_batch()` 按 `len(input_ids)` 计算 seq_len，并 **assert** 每个请求的 image placeholder 数量等于 `image_token_count`。
3. `positions` 从 0 到 `total_seq_len - 1` 连续编号，image tokens 占用连续位置。
4. `build_prefill_inputs()` 返回 `{"pixel_values": ...}`，或让 `PrefillExecutor` 自己从 `req_dicts` 拿 `multi_modal_inputs`。

建议：**image embeddings 在 `PrefillExecutor` 内部计算，不通过 `InputBatchBuilder` 传递 embeddings**。`InputBatchBuilder` 只负责：

- 按已展开的 `input_ids` 计算 seq_len / positions / slot_mapping；
- 把 `has_multimodal_requests` 等标志写进 `attn_metadata`；
- 通过 `req_dicts` 让 executor 拿到 pixel_values。

image tokens 对 KV cache、attention、slot_mapping 来说都是普通 token，只是它们的初始 hidden states 来自 vision encoder 而不是 text embedding。

### 3.7 PrefillExecutor 改造

```python
class PrefillExecutor:
    def execute(self, request_ids, scheduler, chunk_len):
        ...
        mm_inputs = self.multimodal_processor.build_prefill_inputs(req_dicts_prefill)
        if mm_inputs:
            mm_embeddings = self.model.get_multimodal_embeddings(**mm_inputs)
            curr_input = self._inject_multimodal_embeddings(
                curr_input, req_dicts_prefill, mm_embeddings
            )
        logits = self.model(curr_input, positions, kv_caches, attn_metadata)
        ...
```

`_inject_multimodal_embeddings()`：

- 根据每个请求的 `image_token_count` 和 placeholder token id，把 image embeddings 插入到 `curr_input` 中。
- 插入后 `curr_input` 不再是 int token ids，而是 embeddings（因为 model forward 在 `input_ids.dtype == torch.long` 时做 embedding，否则直接用）。

**语义切换**：旧的 `LiteMultiModalProcessor.get_multimodal_embeddings()` 会把 image embeddings 聚合成 request-level vector 再加到所有 token 上。Phase 2A 开始改为 **按 placeholder token 位置替换 embedding**。旧逻辑要删除或移到参考路径，不能再被主路径调用。

### 3.8 Continuous batching

Phase 2A 不支持 continuous batching 或 mixed multimodal batch。一个 prefill batch 内：

- 只能有一个请求；
- 该请求只能有一张图片；
- 所有请求要么都有图、要么都没图（按此分组）。

Phase 2B 再支持：

- 多个请求同时有图片；
- 每请求多张图片；
- 有图/无图请求 mixed batch。

### 3.9 Multimodal + LoRA

Phase 1/2A/2B/3 都不支持。Phase 4 再考虑：

- LoRA 可以作用于 vision projector / connector；
- `is_multimodal_lora` 标志已经存在；
- 需要把 multimodal 输入也纳入 LoRA 调度约束。

---

## 4. 接口与模块变更

### 4.1 新增文件

- `vllm/lora/loader.py`: LoRA checkpoint 加载。
- `vllm/lora/manager.py`: `LoRAManager`，管理 adapter weights 并计算 delta。
- `vllm/lora/weights.py`: `LoRALayerWeights`。
- `vllm/model_executor/models/vision_utils.py`: image 预处理工具（resize / normalize / token count）。
- `vllm/model_executor/models/gemma4/vision.py`: Gemma4 vision tower + connector（Phase 2A）。

### 4.2 修改文件

- `vllm/model_executor/layers/lite_linear.py`: 接入 `LoRAManager`，forward 中计算 LoRA delta。
- `vllm/engine/lite_engine.py`: 初始化 `LoRAManager`；在 `register_lora_adapter` 时加载 weights。
- `vllm/engine/multimodal_processor.py`: 新增 `prepare_before_tokenize()`，Phase 2A 只保留单请求图片处理。
- `vllm/engine/input_batch_builder.py`: 按已展开 input_ids 计算 seq_len/positions，校验 placeholder count。
- `vllm/engine/prefill_executor.py`: 调用 `get_multimodal_embeddings` 并注入 embeddings。
- `vllm/model_executor/models/gemma4/model.py`: 删除 text-only 限制，接入 vision tower。
- `vllm/engine/request_state.py`: 增加 `image_token_count` 字段（或从 `multi_modal_inputs` 派生）。

### 4.3 能力判断

在 `ModelCapabilities` 里新增两个字段（带默认值，不影响旧 adapter）：

```python
@dataclass(frozen=True)
class ModelCapabilities:
    ...
    supports_lora: bool = False
    supports_multimodal: bool = False
```

各 adapter 的 `detect()` 按模型类型设置：

- `llama`, `qwen3_5`, `gemma4`: `supports_lora=True`。
- `gemma4`（Phase 2A 后）: `supports_multimodal=True`。
- `qwen2_vl`（Phase 2B 后）: `supports_multimodal=True`。
- `deepseek_v4_flash`: 两者都 `False`。

`LiteEngine.add_request()` 里：

- 如果 `lora_request` 非空但 `model_capabilities.supports_lora` 为 False，直接拒绝。
- 如果 `multi_modal_data` 非空但 `model_capabilities.supports_multimodal` 为 False，直接拒绝。

---

## 5. 阶段划分

### Phase 1：LoRA 基础（单 adapter per batch）

- [ ] 新增 `LoRALayerWeights`、`LoRALoader`（仅 Safetensors）、`LoRAManager`。
- [ ] `LiteLinear.forward()` 接入 LoRA delta 计算。
- [ ] `LoRAManager.bind_to_model()` 给每个 `LiteLinear` 注入 manager 和 target_name。
- [ ] `LiteEngine.register_lora_adapter()` 加载 weights 到 `LoRAManager`。
- [ ] `ModelCapabilities` 加 `supports_lora`，Llama/Qwen3.5/Gemma4 adapter 设为 True；DeepSeek False。
- [ ] Llama / Qwen3.5 / Gemma4 模型层自动生效（因为都走 `LiteLinear`）。
- [ ] `StepScheduler` 按 `lora_id` 预分组 batch；`InputBatchBuilder` 只做 mixed LoRA 防御性 assert。
- [ ] 单 adapter LoRA 单测 + Llama/Gemma4 regression。

### Phase 2A：Multimodal 基础 — 单请求单图 + Gemma4（image only）

- [ ] 新增 `LiteMultiModalProcessor.prepare_before_tokenize()`，在 `LiteRequestBuilder` encode 前完成图片加载和 `image_token_count` 计算。
- [ ] `LiteRequestBuilder` 在 encode 前用 `image_token_count` 把 prompt 中的 `<image>` 展开成 N 个 `image_token_id`。
- [ ] `RequestState` 增加 `image_token_count`。
- [ ] `InputBatchBuilder` 校验 image placeholder 数量。
- [ ] 实现 Gemma4 vision tower + connector，删除 text-only 限制。
- [ ] `PrefillExecutor` 按 placeholder 位置注入 image embeddings（删除旧的 request-level 聚合逻辑）。
- [ ] Gemma4 单请求单图回归测试。

### Phase 2B：Multimodal 扩展 — 多图、多请求 continuous batching、Qwen2VL

- [ ] 扩展 `LiteMultiModalProcessor` 支持多请求 image batch。
- [ ] 支持每请求多张图片。
- [ ] 实现 Qwen2VL 真实 vision tower + projector（动态分辨率）。
- [ ] `PrefillExecutor` 支持多请求/多图 embedding 注入。
- [ ] 图文混合 continuous batching 单测。

### Phase 3：Mixed LoRA batch

- [ ] `LoRAManager` 支持 stacked weights + per-token/per-request index mapping。
- [ ] 升级 `lora_mapping` 为 `LoRAMapping` dataclass。
- [ ] `StepScheduler` 允许同一 batch 内不同 LoRA。
- [ ] mixed LoRA 单测。

### Phase 4：Multimodal + LoRA

- [ ] LoRA 作用于 vision projector / connector。
- [ ] 调度约束同时考虑 multimodal 和 LoRA。

---

## 6. 测试计划

### 6.1 LoRA 测试

- `tests/lora/test_loader.py`: 加载 PEFT checkpoint，验证 key 映射、rank/alpha。
- `tests/lora/test_manager.py`: 单 adapter `compute_delta`，无 LoRA 返回 None。
- `tests/lora/test_lite_linear.py`: `LiteLinear.forward` 在 LoRA active 时输出 = base + delta。
- `tests/lora/test_mixed_rejected.py`: `InputBatchBuilder` 对 mixed LoRA batch 报错。
- 集成：`LLM.generate()` 带 `lora_request`，输出和 base 模型不同但合理。

### 6.2 Multimodal Phase 2A 测试

- `tests/multimodal/test_image_preprocessing.py`: resize/normalize/token count。
- `tests/multimodal/test_request_builder_image_tokens.py`: `<image>` 被展开成 N 个 placeholder token。
- `tests/multimodal/test_input_batch_builder.py`: image placeholder 数量等于 `image_token_count`。
- `tests/multimodal/test_gemma4_vision.py`: vision tower + connector 输出 shape 正确。
- `tests/multimodal/test_embedding_injection.py`: image embeddings 正确替换 placeholder 位置。
- 集成：Gemma4 单请求单图 prompt 生成文本。

### 6.3 Multimodal Phase 2B 测试

- `tests/multimodal/test_multimodal_processor_batch.py`: 多请求 pixel_values 聚合。
- `tests/multimodal/test_multi_image_request.py`: 每请求多张图片。
- `tests/multimodal/test_qwen2vl_vision.py`: Qwen2VL 真实 vision tower。
- `tests/multimodal/test_mixed_multimodal_batch.py`: 有图/无图请求 continuous batching。
- 集成：Gemma4 多图 + 图文混合 batch。

---

## 7. 风险与缓解

| 风险 | 缓解 |
|------|------|
| LoRA 加载时 PEFT key 和模型 prefix 对不上 | 维护一个映射表，单测覆盖常见 checkpoint 格式 |
| LoRA weight 形状和 PEFT 保存不一致导致 silent wrong | Loader 转置并 assert shape，单测覆盖 |
| LoRA delta 与量化输出 dtype 不一致 | delta 计算后 cast 到 base_out dtype 再相加 |
| Mixed LoRA batch 过早引入导致复杂度爆炸 | Phase 1 StepScheduler 预分组，禁止 mixed batch |
| Gemma4 vision tower 权重加载复杂 | 复用 HF `AutoModel` 加载 vision tower，只冻结推理 |
| Image token 数量双算导致 seq_len 错误 | 统一由 request builder 展开 placeholder，InputBatchBuilder 只校验 |
| 旧的 request-level image embedding 聚合逻辑和新 placeholder 替换冲突 | Phase 2A 删除/隔离旧聚合逻辑 |
| Multimodal + chunked prefill 交互复杂 | Phase 2A 不允许对 multimodal 请求做 chunk prefill（prefill 必须一次完成） |
| DeepSeek 自定义路径误受影响 | DeepSeek adapter 继续拒绝 LoRA/multimodal |

---

## 8. 结论

- **LoRA 先做多，multimodal 后做**：LoRA 的基础设施已经基本就绪，只是缺 weight 加载和应用；multimodal 还需要新增 vision tower。
- **两者都分阶段**：先支持基础场景，再支持混合 batch / multimodal LoRA。
- **DeepSeek 不纳入范围**：当前聚焦 LiteEngine 标准模型。
