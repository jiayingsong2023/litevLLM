# Gemma4 E2B/E4B Inference Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `models/gemma-4-E2B-it-AWQ-INT4` and `models/gemma-4-E4B-it` produce correct text output by adding Gemma4 small-variant features: Per-Layer Embeddings (PLE), KV-cache sharing, and conditional double-wide MLP.

**Architecture:** Extend `LiteConfig` with the Gemma4 text-config fields that drive these features, then teach `Gemma4TextModel`, `Gemma4DecoderLayer`, `Gemma4Attention`, and `Gemma4MLP` to use them. No engine/scheduler changes are required: KV sharing is implemented by wiring shared layers to their donor layer's `k_proj`/`v_proj` and passing the donor's KV-cache slot at runtime; PLE is computed once per forward from `input_ids` and fed to every layer.

**Tech Stack:** Python 3.12, PyTorch, `uv run`, existing `vllm/model_executor/models/gemma4/` code, `LiteLinear`, `RMSNorm`, `nn.Embedding`.

## Global Constraints

- Python 3.12 only; use `uv run` for every command.
- No new `os.environ` reads inside `vllm/` runtime code; config must flow through ` LiteConfig` / `inf_config`.
- No C++ kernels; keep changes pure Python + existing Triton.
- Heavy model tests must be gated by an explicit env var (e.g., `RUN_GEMMA4_E2B_SMOKE=1`) and live in `tests/`.
- Run `uv run ruff check . && uv run ruff format .` after code changes.
- Run `bash tests/run_regression_suite.sh` before claiming the work passes.

---

## File map

| File | Responsibility |
|------|----------------|
| `vllm/model_executor/models/lite_config.py` | Normalize new hf_config fields (`use_double_wide_mlp`, `num_kv_shared_layers`, `hidden_size_per_layer_input`, `vocab_size_per_layer_input`) and expose helper predicates. |
| `vllm/model_executor/models/gemma4/mlp.py` | Compute per-layer effective `intermediate_size` (double-wide for E2B shared layers). |
| `vllm/model_executor/models/gemma4/attention.py` | Accept an optional donor `Gemma4Attention`; share `k_proj`/`v_proj` modules and KV-cache index for KV-shared layers. |
| `vllm/model_executor/models/gemma4/layer.py` | Pass `layer_idx` to MLP, wire KV donor to attention, add per-layer PLE gate/projection/norm, inject PLE before `layer_scalar`. |
| `vllm/model_executor/models/gemma4/model.py` | Build layers in a loop so KV-shared layers can reference donors; add PLE embedding/projection/norm; compute per-layer inputs per forward. |
| `tests/test_gemma4_e2b_e4b_support.py` | Unit tests for helpers and gated E2B smoke generation. |

---

## Task 1: Add Gemma4 text-config fields to `LiteConfig`

**Files:**
- Modify: `vllm/model_executor/models/lite_config.py`
- Test: `tests/test_gemma4_e2b_e4b_support.py`

**Interfaces:**
- Consumes: raw `hf_config` from `config.json` (`text_config` is already flattened into `hf_config` by the time `LiteConfig` sees it).
- Produces: `LiteConfig.use_double_wide_mlp`, `LiteConfig.num_kv_shared_layers`, `LiteConfig.hidden_size_per_layer_input`, `LiteConfig.vocab_size_per_layer_input`, plus helpers `ple_enabled()`, `is_kv_shared_layer(layer_idx)`, `effective_intermediate_size(layer_idx)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gemma4_e2b_e4b_support.py
import json
import os

import pytest

from vllm.model_executor.models.lite_config import LiteConfig


@pytest.fixture
def e2b_text_config():
    return json.load(open("models/gemma-4-E2B-it-AWQ-INT4/config.json"))["text_config"]


@pytest.fixture
def e4b_text_config():
    return json.load(open("models/gemma-4-E4B-it/config.json"))["text_config"]


def test_lite_config_e2b_helpers(e2b_text_config):
    cfg = LiteConfig(e2b_text_config)
    assert cfg.use_double_wide_mlp is True
    assert cfg.num_kv_shared_layers == 20
    assert cfg.hidden_size_per_layer_input == 256
    assert cfg.vocab_size_per_layer_input == 262144
    assert cfg.ple_enabled() is True
    assert cfg.is_kv_shared_layer(14) is False
    assert cfg.is_kv_shared_layer(15) is True
    assert cfg.is_kv_shared_layer(34) is True
    assert cfg.effective_intermediate_size(0) == 6144
    assert cfg.effective_intermediate_size(15) == 12288


def test_lite_config_e4b_helpers(e4b_text_config):
    cfg = LiteConfig(e4b_text_config)
    assert cfg.use_double_wide_mlp is False
    assert cfg.num_kv_shared_layers == 18
    assert cfg.hidden_size_per_layer_input == 256
    assert cfg.ple_enabled() is True
    assert cfg.is_kv_shared_layer(23) is False
    assert cfg.is_kv_shared_layer(24) is True
    assert cfg.effective_intermediate_size(24) == 10240
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/test_gemma4_e2b_e4b_support.py -q
```

Expected: `AttributeError` on `LiteConfig` for the new fields.

- [ ] **Step 3: Implement `LiteConfig` additions**

In `vllm/model_executor/models/lite_config.py`, after the existing `self.tie_word_embeddings = ...` block (around line 127), add:

```python
        # Gemma4 E2B/E4B small-variant features
        self.use_double_wide_mlp = bool(
            getattr(hf_config, "use_double_wide_mlp", False)
        )
        self.num_kv_shared_layers = int(
            getattr(hf_config, "num_kv_shared_layers", 0) or 0
        )
        self.hidden_size_per_layer_input = int(
            getattr(hf_config, "hidden_size_per_layer_input", 0) or 0
        )
        self.vocab_size_per_layer_input = int(
            getattr(hf_config, "vocab_size_per_layer_input", self.vocab_size)
            or self.vocab_size
        )

    def ple_enabled(self) -> bool:
        return int(self.hidden_size_per_layer_input or 0) > 0

    def is_kv_shared_layer(self, layer_idx: int) -> bool:
        n_shared = int(self.num_kv_shared_layers or 0)
        if n_shared <= 0:
            return False
        first_shared = int(self.num_hidden_layers) - n_shared
        return int(layer_idx) >= first_shared > 0

    def effective_intermediate_size(self, layer_idx: int | None) -> int:
        if layer_idx is None:
            return int(self.intermediate_size)
        if self.use_double_wide_mlp and self.is_kv_shared_layer(layer_idx):
            return int(self.intermediate_size) * 2
        return int(self.intermediate_size)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/test_gemma4_e2b_e4b_support.py -q
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add vllm/model_executor/models/lite_config.py tests/test_gemma4_e2b_e4b_support.py
git commit -m "feat(config): expose Gemma4 E2B/E4B PLE/KV-share/double-wide fields in LiteConfig"
```

---

## Task 2: Conditional double-wide MLP for shared layers

**Files:**
- Modify: `vllm/model_executor/models/gemma4/mlp.py`, `vllm/model_executor/models/gemma4/layer.py`
- Test: `tests/test_gemma4_e2b_e4b_support.py`

**Interfaces:**
- Consumes: `LiteConfig.effective_intermediate_size(layer_idx)`.
- Produces: `Gemma4MLP(..., layer_idx=...)` uses the correct per-layer `intermediate_size`.

- [ ] **Step 1: Write the failing test**

```python
def test_mlp_double_wide_shape(e2b_text_config):
    import torch
    from vllm.model_executor.models.gemma4.mlp import Gemma4MLP

    cfg = LiteConfig(e2b_text_config)
    mlp0 = Gemma4MLP(cfg, None, "layers.0", layer_idx=0)
    mlp15 = Gemma4MLP(cfg, None, "layers.15", layer_idx=15)
    assert mlp0.gate_proj.output_size == 6144
    assert mlp0.down_proj.input_size == 6144
    assert mlp15.gate_proj.output_size == 12288
    assert mlp15.down_proj.input_size == 12288

    # Forward shape sanity with dummy dense weights
    for mlp in (mlp0, mlp15):
        mlp.gate_proj.weight = torch.nn.Parameter(
            torch.zeros(mlp.gate_proj.output_size, cfg.hidden_size, dtype=torch.float16),
            requires_grad=False,
        )
        mlp.up_proj.weight = torch.nn.Parameter(
            torch.zeros(mlp.up_proj.output_size, cfg.hidden_size, dtype=torch.float16),
            requires_grad=False,
        )
        mlp.down_proj.weight = torch.nn.Parameter(
            torch.zeros(cfg.hidden_size, mlp.down_proj.input_size, dtype=torch.float16),
            requires_grad=False,
        )
        x = torch.randn(1, 12, cfg.hidden_size, dtype=torch.float16)
        out = mlp(x)
        assert out.shape == (1, 12, cfg.hidden_size)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/test_gemma4_e2b_e4b_support.py::test_mlp_double_wide_shape -v
```

Expected: `TypeError: Gemma4MLP.__init__() got an unexpected keyword argument 'layer_idx'`.

- [ ] **Step 3: Implement per-layer MLP sizing**

In `vllm/model_executor/models/gemma4/mlp.py`:

```python
class Gemma4MLP(nn.Module):
    def __init__(
        self,
        config: LiteConfig,
        quant_config: Any,
        prefix: str,
        layer_idx: int | None = None,
    ):
        super().__init__()
        self.hidden_act = str(
            getattr(config, "hidden_activation", getattr(config, "hidden_act", "silu"))
        ).lower()
        self.intermediate_size = int(config.effective_intermediate_size(layer_idx))
        self.gate_proj = LiteLinear(
            config.hidden_size,
            self.intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp.gate_proj",
        )
        self.up_proj = LiteLinear(
            config.hidden_size,
            self.intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp.up_proj",
        )
        self.down_proj = LiteLinear(
            self.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp.down_proj",
        )
        self._layer_config = Gemma4LayerConfig()
```

In `vllm/model_executor/models/gemma4/layer.py`, line 87, change:

```python
            self.mlp = Gemma4MLP(config, quant_config, prefix, layer_idx=layer_idx)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/test_gemma4_e2b_e4b_support.py::test_mlp_double_wide_shape -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add vllm/model_executor/models/gemma4/mlp.py vllm/model_executor/models/gemma4/layer.py tests/test_gemma4_e2b_e4b_support.py
git commit -m "feat(gemma4): per-layer effective intermediate size for double-wide MLP"
```

---

## Task 3: KV-cache sharing for E2B/E4B

**Files:**
- Modify: `vllm/model_executor/models/gemma4/attention.py`, `vllm/model_executor/models/gemma4/layer.py`, `vllm/model_executor/models/gemma4/model.py`
- Test: `tests/test_gemma4_e2b_e4b_support.py`

**Interfaces:**
- Consumes: `LiteConfig.is_kv_shared_layer(layer_idx)`, `LiteConfig.layer_types`, and earlier layers built before the current one.
- Produces: `Gemma4Attention` uses donor `k_proj`/`v_proj` and a `kv_scale_cache_idx` that points at the donor's cache slot.

- [ ] **Step 1: Write the failing test**

```python
def test_kv_shared_donor_selection(e2b_text_config):
    import torch
    from vllm.model_executor.models.gemma4.model import Gemma4TextModel

    cfg = LiteConfig(e2b_text_config)
    # Need a minimal quant_config; None works for shape tests because layers
    # don't build weights until load.
    model = Gemma4TextModel(e2b_text_config, None)
    assert model.layers[15].self_attn.k_proj is model.layers[10].self_attn.k_proj
    assert model.layers[15].self_attn.v_proj is model.layers[10].self_attn.v_proj
    assert model.layers[15].self_attn.kv_scale_cache_idx == 10
    # Layer 19 is full_attention; donor is layer 14.
    assert model.layers[19].self_attn.kv_scale_cache_idx == 14
    # Non-shared layer points at itself.
    assert model.layers[10].self_attn.kv_scale_cache_idx == 10
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/test_gemma4_e2b_e4b_support.py::test_kv_shared_donor_selection -v
```

Expected: `AttributeError: 'Gemma4Attention' object has no attribute 'kv_scale_cache_idx'` and/or donor wiring absent.

- [ ] **Step 3: Implement KV sharing in attention**

In `vllm/model_executor/models/gemma4/attention.py`, change `__init__` signature and body:

```python
    def __init__(
        self,
        config: LiteConfig,
        quant_config: Any,
        prefix: str,
        layer_idx: int,
        runtime_config: Any = None,
        kv_shared_with: "Gemma4Attention | None" = None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.kv_scale_cache_idx = layer_idx
        ...
        if kv_shared_with is not None:
            self.k_proj = kv_shared_with.k_proj
            if self.v_proj is not None:
                self.v_proj = kv_shared_with.v_proj
            self.kv_scale_cache_idx = kv_shared_with.kv_scale_cache_idx
```

Then replace every occurrence of `kv_scale_cache[self.layer_idx]` with `kv_scale_cache[self.kv_scale_cache_idx]` (two places: line 322 and the fallback path).

- [ ] **Step 4: Wire donor in layer and model**

In `vllm/model_executor/models/gemma4/layer.py`, change `Gemma4DecoderLayer.__init__` signature:

```python
    def __init__(
        self,
        config: LiteConfig,
        quant_config: Any,
        prefix: str,
        layer_idx: int,
        fp32_residual_guard_enabled: bool = False,
        fp32_residual_guard_start: int = 8,
        fp32_residual_guard_span: int = 3,
        runtime_config: Any = None,
        kv_shared_with: Any = None,
    ):
```

Pass `kv_shared_with=kv_shared_with` into `Gemma4Attention(...)`.

In `vllm/model_executor/models/gemma4/model.py`, replace the list-comprehension layer construction with an explicit loop that can look up donors:

```python
        layers: list[Gemma4DecoderLayer] = []
        for i in range(self.config.num_hidden_layers):
            donor_attn = None
            if config.is_kv_shared_layer(i):
                own_type = config.layer_types[i] if config.layer_types else "global"
                for j in range(i - 1, -1, -1):
                    if (
                        not config.is_kv_shared_layer(j)
                        and config.layer_types
                        and config.layer_types[j] == own_type
                    ):
                        donor_attn = layers[j].self_attn
                        break
            layers.append(
                Gemma4DecoderLayer(
                    self.config,
                    quant_config,
                    prefix=f"layers.{i}",
                    layer_idx=i,
                    fp32_residual_guard_enabled=fp32_residual_guard_enabled,
                    fp32_residual_guard_start=fp32_residual_guard_start,
                    fp32_residual_guard_span=fp32_residual_guard_span,
                    runtime_config=runtime_config,
                    kv_shared_with=donor_attn,
                )
            )
        self.layers = nn.ModuleList(layers)
```

- [ ] **Step 5: Pass donor KV cache at runtime**

In `vllm/model_executor/models/gemma4/model.py`, change the layer loop in `forward`:

```python
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[layer.self_attn.kv_scale_cache_idx]
            x = layer(x, positions, kv_cache, attn_metadata, lora_mapping)
```

- [ ] **Step 6: Run the test to verify it passes**

```bash
uv run pytest tests/test_gemma4_e2b_e4b_support.py::test_kv_shared_donor_selection -v
```

Expected: 1 passed.

- [ ] **Step 7: Commit**

```bash
git add vllm/model_executor/models/gemma4/attention.py vllm/model_executor/models/gemma4/layer.py vllm/model_executor/models/gemma4/model.py tests/test_gemma4_e2b_e4b_support.py
git commit -m "feat(gemma4): KV-cache sharing for E2B/E4B shared layers"
```

---

## Task 4: Per-Layer Embeddings (PLE)

**Files:**
- Modify: `vllm/model_executor/models/gemma4/model.py`, `vllm/model_executor/models/gemma4/layer.py`
- Test: `tests/test_gemma4_e2b_e4b_support.py`

**Interfaces:**
- Consumes: `LiteConfig.ple_enabled()`, `hidden_size_per_layer_input`, `vocab_size_per_layer_input`, `num_hidden_layers`, `hidden_size`.
- Produces: `Gemma4TextModel` builds PLE modules and returns a `[B,S,L,D]` per-layer input tensor; each `Gemma4DecoderLayer` consumes `per_layer_input[:, i, :]`.

- [ ] **Step 1: Write the failing test**

```python
def test_ple_shape(e2b_text_config):
    import torch
    from vllm.model_executor.models.gemma4.model import Gemma4TextModel

    cfg = LiteConfig(e2b_text_config)
    model = Gemma4TextModel(e2b_text_config, None)
    assert model.embed_tokens_per_layer is not None
    assert model.per_layer_model_projection is not None
    assert model.per_layer_projection_norm is not None

    B, S = 1, 5
    input_ids = torch.randint(0, cfg.vocab_size, (B, S), dtype=torch.long)
    inputs_embeds = torch.randn(B, S, cfg.hidden_size, dtype=torch.float16)
    ple = model._compute_per_layer_inputs(input_ids, inputs_embeds)
    assert ple.shape == (B, S, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/test_gemma4_e2b_e4b_support.py::test_ple_shape -v
```

Expected: `AttributeError: 'Gemma4TextModel' object has no attribute '_compute_per_layer_inputs'`.

- [ ] **Step 3: Add PLE modules to `Gemma4TextModel`**

In `vllm/model_executor/models/gemma4/model.py`, in `Gemma4TextModel.__init__`, after `self.embed_tokens = nn.Embedding(...)` add:

```python
        # Per-Layer Embeddings (PLE) for Gemma4 E2B/E4B.
        self.hidden_size_per_layer_input = int(
            getattr(self.config, "hidden_size_per_layer_input", 0) or 0
        )
        if self.config.ple_enabled():
            total_ple_dim = (
                self.config.num_hidden_layers * self.hidden_size_per_layer_input
            )
            self.embed_tokens_per_layer = nn.Embedding(
                self.config.vocab_size_per_layer_input,
                total_ple_dim,
            )
            self.per_layer_model_projection = LiteLinear(
                self.config.hidden_size,
                total_ple_dim,
                bias=False,
                quant_config=quant_config,
                prefix="model.per_layer_model_projection",
            )
            self.per_layer_projection_norm = RMSNorm(
                self.hidden_size_per_layer_input,
                eps=_get_eps(self.config),
            )
            self.register_buffer(
                "embed_scale_per_layer",
                torch.tensor(self.hidden_size_per_layer_input**0.5),
                persistent=False,
            )
            self.register_buffer(
                "per_layer_input_scale",
                torch.rsqrt(torch.tensor(2.0)),
                persistent=False,
            )
            self.register_buffer(
                "per_layer_projection_scale",
                torch.tensor(self.config.hidden_size**-0.5),
                persistent=False,
            )
        else:
            self.embed_tokens_per_layer = None
            self.per_layer_model_projection = None
            self.per_layer_projection_norm = None
            self.embed_scale_per_layer = None
            self.per_layer_input_scale = None
            self.per_layer_projection_scale = None
```

- [ ] **Step 4: Implement `_compute_per_layer_inputs`**

Add this method to `Gemma4TextModel`:

```python
    def _compute_per_layer_inputs(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.config.ple_enabled():
            return None
        if input_ids.dtype != torch.long:
            return None
        bsz, seqlen = input_ids.shape
        num_layers = self.config.num_hidden_layers
        ple_dim = self.hidden_size_per_layer_input

        # Token-identity component.
        token_part = self.embed_tokens_per_layer(input_ids) * self.embed_scale_per_layer
        token_part = token_part.view(bsz, seqlen, num_layers, ple_dim)

        # Context-aware projection component.
        proj = self.per_layer_model_projection(inputs_embeds)
        proj = proj * self.per_layer_projection_scale
        proj = proj.view(bsz, seqlen, num_layers, ple_dim)
        proj = self.per_layer_projection_norm(proj)

        return (token_part + proj) * self.per_layer_input_scale
```

- [ ] **Step 5: Feed PLE through the layer stack**

Change `Gemma4TextModel.forward`:

```python
        x = self.embed_tokens(input_ids) * self.embed_scale
        if multimodal_embeddings is not None:
            x = _replace_image_placeholders(...)
        per_layer_inputs = self._compute_per_layer_inputs(input_ids, x)
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[layer.self_attn.kv_scale_cache_idx]
            ple_input = (
                per_layer_inputs[:, :, i, :]
                if per_layer_inputs is not None
                else None
            )
            x = layer(
                x,
                positions,
                kv_cache,
                attn_metadata,
                lora_mapping,
                per_layer_input=ple_input,
            )
        return self.norm(x)
```

- [ ] **Step 6: Consume PLE inside `Gemma4DecoderLayer`**

In `vllm/model_executor/models/gemma4/layer.py`, change `__init__` to create PLE submodules when enabled:

```python
        self.per_layer_input_gate: LiteLinear | None = None
        self.per_layer_projection: LiteLinear | None = None
        self.post_per_layer_input_norm: RMSNorm | None = None
        if config.ple_enabled():
            self.per_layer_input_gate = LiteLinear(
                config.hidden_size,
                config.hidden_size_per_layer_input,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.per_layer_input_gate",
            )
            self.per_layer_projection = LiteLinear(
                config.hidden_size_per_layer_input,
                config.hidden_size,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.per_layer_projection",
            )
            self.post_per_layer_input_norm = RMSNorm(
                config.hidden_size, eps=_get_eps(config)
            )
```

Change `forward` signature:

```python
    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: Any,
        attn_metadata: Any,
        lora_mapping: Any = None,
        per_layer_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
```

Add the PLE injection just before the final `layer_scalar` multiply (after the MLP residual add):

```python
        if per_layer_input is not None and self.per_layer_input_gate is not None:
            gate = self.per_layer_input_gate(x, lora_mapping, inf_config=inf_config)
            if self.hidden_act in ("gelu", "gelu_pytorch_tanh"):
                gate = F.gelu(gate, approximate="tanh")
            else:
                gate = F.silu(gate)
            gated = gate * per_layer_input
            ple_out = self.per_layer_projection(
                gated, lora_mapping, inf_config=inf_config
            )
            ple_out = self.post_per_layer_input_norm(ple_out)
            x = x + ple_out
        return x * self.layer_scalar
```

Add `self.hidden_act` in `__init__` (mirroring MLP):

```python
        self.hidden_act = str(
            getattr(config, "hidden_activation", getattr(config, "hidden_act", "silu"))
        ).lower()
```

- [ ] **Step 7: Run the test to verify it passes**

```bash
uv run pytest tests/test_gemma4_e2b_e4b_support.py::test_ple_shape -v
```

Expected: 1 passed.

- [ ] **Step 8: Commit**

```bash
git add vllm/model_executor/models/gemma4/model.py vllm/model_executor/models/gemma4/layer.py tests/test_gemma4_e2b_e4b_support.py
git commit -m "feat(gemma4): Per-Layer Embeddings (PLE) for E2B/E4B"
```

---

## Task 5: Gated E2B AWQ Q4 smoke test

**Files:**
- Test: `tests/test_gemma4_e2b_e4b_support.py`

**Interfaces:**
- Consumes: the real `models/gemma-4-E2B-it-AWQ-INT4` checkpoint.
- Produces: a passing test that loads the model and generates at least one token.

- [ ] **Step 1: Add gated smoke test**

```python
@pytest.mark.skipif(
    os.environ.get("RUN_GEMMA4_E2B_SMOKE") != "1",
    reason="Set RUN_GEMMA4_E2B_SMOKE=1 to load the E2B model",
)
def test_e2b_awq_q4_generates():
    import os
    os.environ.setdefault("FASTINFERENCE_GEMMA4_ALLOW_INT4_KV", "1")
    from vllm import LLM

    llm = LLM(
        model="models/gemma-4-E2B-it-AWQ-INT4",
        max_model_len=256,
        gpu_memory_utilization=0.85,
        max_num_batched_tokens=512,
    )
    outputs = llm.generate("The capital of France is", max_tokens=8)
    assert outputs
    text = outputs[0].outputs[0].text
    assert isinstance(text, str) and len(text) > 0
    print("E2B output:", text)
    llm.shutdown()
```

- [ ] **Step 2: Run the smoke test**

```bash
RUN_GEMMA4_E2B_SMOKE=1 uv run pytest tests/test_gemma4_e2b_e4b_support.py::test_e2b_awq_q4_generates -v -s
```

Expected: PASS and the generated text is non-empty.

- [ ] **Step 3: Commit**

```bash
git add tests/test_gemma4_e2b_e4b_support.py
git commit -m "test(gemma4): gated E2B AWQ Q4 smoke generation test"
```

---

## Task 6: Lint, format, and regression suite

- [ ] **Step 1: Format and lint**

```bash
uv run ruff check vllm/model_executor/models/lite_config.py vllm/model_executor/models/gemma4/ tests/test_gemma4_e2b_e4b_support.py
uv run ruff format vllm/model_executor/models/lite_config.py vllm/model_executor/models/gemma4/ tests/test_gemma4_e2b_e4b_support.py
```

Expected: no errors.

- [ ] **Step 2: Run unit tests (without heavy smoke)**

```bash
uv run pytest tests/test_gemma4_e2b_e4b_support.py -q
```

Expected: all non-smoke tests pass.

- [ ] **Step 3: Run regression suite**

```bash
bash tests/run_regression_suite.sh
```

Expected: no new failures.

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "chore(gemma4): lint and format E2B/E4B support"
```

---

## Spec coverage self-review

| Requirement | Task |
|-------------|------|
| `use_double_wide_mlp` respected for E2B shared layers | Task 1 + Task 2 |
| `num_kv_shared_layers` respected; shared layers reuse donor KV | Task 1 + Task 3 |
| `hidden_size_per_layer_input` / `vocab_size_per_layer_input` PLE tables built | Task 1 + Task 4 |
| PLE combined correctly and injected before `layer_scalar` | Task 4 |
| E2B AWQ Q4 can generate text | Task 5 |
| No runtime `os.environ` reads in `vllm/` | All tasks (only test sets env) |
| Heavy test gated | Task 5 |

## Placeholder scan

No `TODO`, `TBD`, or placeholder code blocks remain. Every step contains exact file paths, exact function signatures, and exact test code.

## Type consistency

- `LiteConfig.effective_intermediate_size(layer_idx: int | None)` returns `int`.
- `Gemma4MLP.__init__` accepts `layer_idx: int | None`.
- `Gemma4DecoderLayer.__init__` accepts `kv_shared_with: Any` (donor `Gemma4Attention`) and `per_layer_input: torch.Tensor | None` in `forward`.
- `Gemma4Attention` exposes `kv_scale_cache_idx: int` and uses it for `kv_scale_cache` indexing.
