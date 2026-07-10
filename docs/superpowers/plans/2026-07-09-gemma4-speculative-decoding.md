# Gemma4 Speculative Decoding (P1 + P1.5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver offline P1 draft-model acceptance measurement and a P1.5 cached-verifier microbenchmark that can predict whether speculative decoding will speed up Gemma4-26B decode. Stop if gates fail.

**Architecture:** Extend the existing `tests/tools/gemma4_speculative_*.py` offline tools; add a minimal verifier flag to `Gemma4ForConditionalGeneration.forward`; build a standalone cached-verifier runner that reuses an existing request's KV blocks and actually mutates seq_len/block tables during a multi-step speculative simulation; run a fixed prompt set with real E2B/E4B draft models and gate on bit-exact correctness plus predicted decode TPS.

**Tech Stack:** Python 3.12, `uv run`, PyTorch, existing FastInference lite engine (`vllm/`), pytest.

## Global Constraints

- Python 3.12 only; use `uv run` for every Python command.
- No runtime `os.environ` reads in `vllm/` production code.
- P1/P1.5 code lives in `tests/tools/` except for the minimal verifier flag in `vllm/model_executor/models/gemma4/model.py`.
- No scheduler/backend/API changes in P1/P1.5.
- Bit-exact greedy correctness is a hard gate at every phase.
- All tests must be gated so they do not auto-load large models during normal `pytest` runs.
- Existing uncommitted E2B/E4B support changes in `vllm/` must be committed or checkpointed before speculative tasks begin; this plan does not modify E2B/E4B model loading logic.

---

## File Map

| File | Responsibility |
|---|---|
| `tests/tools/gemma4_speculative_tokenizer_gate.py` | Hardened tokenizer compatibility gate. |
| `tests/test_gemma4_speculative_tokenizer_gate.py` | Unit tests for the gate using fake tokenizers. |
| `tests/tools/fixtures/gemma4_speculative_prompts.json` | Fixed prompt fixture for P1 acceptance sweep and P1.5 microbench. |
| `tests/tools/gemma4_speculative_prototype.py` | P1 offline draft-model prototype; generates acceptance stats. |
| `tests/test_gemma4_speculative_prototype.py` | Mock unit tests for P1 offline prototype. |
| `vllm/model_executor/models/gemma4/model.py` | Add `verifier_return_all_logits` metadata path to forward. |
| `tests/tools/gemma4_speculative_verifier_microbench.py` | P1.5 multi-step cached-verifier microbench. |
| `tests/test_gemma4_speculative_verifier_microbench.py` | Mock unit tests for P1.5 state machine. |

---

### Task 1: Harden Tokenizer Gate

**Files:**
- Modify: `tests/tools/gemma4_speculative_tokenizer_gate.py`
- Modify: `tests/test_gemma4_speculative_tokenizer_gate.py`
- Create: `tests/tools/fixtures/gemma4_speculative_prompts.json`

**Interfaces:**
- Consumes: `get_tokenizer(model_path)` from `vllm.model_executor.model_loader`.
- Produces: `build_report(target, draft, prompts, messages)` returns dict with `passed` bool and diagnostic details.

- [ ] **Step 1: Add full vocabulary mapping check**

```python
def _vocab_hash(tokenizer: Any) -> str:
    vocab = tokenizer.get_vocab()  # dict token -> id
    return hashlib.sha256(
        json.dumps(vocab, sort_keys=True).encode("utf-8")
    ).hexdigest()


def check_full_vocab(target: Any, draft: Any) -> tuple[bool, dict[str, Any]]:
    target_hash = _vocab_hash(target)
    draft_hash = _vocab_hash(draft)
    return target_hash == draft_hash, {
        "target_hash": target_hash,
        "draft_hash": draft_hash,
    }
```

- [ ] **Step 2: Add added-tokens comparison**

```python
def _added_tokens(tokenizer: Any) -> dict[str, Any]:
    added = tokenizer.added_tokens_encoder  # type: ignore[attr-defined]
    return {str(k): int(v) for k, v in (added or {}).items()}


def check_added_tokens(target: Any, draft: Any) -> tuple[bool, dict[str, Any]]:
    target_added = _added_tokens(target)
    draft_added = _added_tokens(draft)
    return target_added == draft_added, {
        "target": target_added,
        "draft": draft_added,
    }
```

- [ ] **Step 3: Add chat-template token-ID comparison as diagnostic**

```python
def check_chat_template_token_ids(
    target: Any, draft: Any, messages: list[dict[str, str]]
) -> tuple[bool | None, dict[str, Any]]:
    target_template = getattr(target, "chat_template", None)
    draft_template = getattr(draft, "chat_template", None)
    if target_template is None or draft_template is None:
        details = {
            "target_has_template": target_template is not None,
            "draft_has_template": draft_template is not None,
            "target_ids": None,
            "draft_ids": None,
        }
        return None, details

    target_text = target.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    draft_text = draft.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    target_ids = target.encode(target_text, add_special_tokens=False)
    draft_ids = draft.encode(draft_text, add_special_tokens=False)
    match = target_ids == draft_ids
    details = {
        "target_text": target_text,
        "draft_text": draft_text,
        "target_ids": target_ids,
        "draft_ids": draft_ids,
    }
    return match, details
```

- [ ] **Step 4: Update `build_report` hard gate**

Hard gate must pass all of: full vocab hash, added tokens, special tokens, direct encode match. Chat-template token-ID match is diagnostic only.

```python
passed = (
    vocab_match
    and added_match
    and special_match
    and encode_match
)
```

- [ ] **Step 5: Update unit tests**

Add `test_full_vocab_mismatch_fails` and `test_added_token_mismatch_fails` in `tests/test_gemma4_speculative_tokenizer_gate.py` using `FakeTokenizer`. Extend `FakeTokenizer` to expose `get_vocab()` and `added_tokens_encoder`.

```python
def __init__(
    self,
    vocab_size: int = 256000,
    bos_token_id: int | list[int] | None = 2,
    eos_token_id: int | list[int] | None = 1,
    pad_token_id: int | list[int] | None = 0,
    encode_map: dict[str, list[int]] | None = None,
    added_tokens_encoder: dict[str, int] | None = None,
    chat_template: str
    | None = "{% for m in messages %}{{ m['role'] }}{% endfor %}",
    chat_output: str = "<chat output>",
) -> None:
    ...
    self.added_tokens_encoder = added_tokens_encoder or {}


def get_vocab(self) -> dict[str, int]:
    return {chr(i): i for i in range(self.vocab_size)}
```

- [ ] **Step 6: Create prompt fixture with real context lengths**

`tests/tools/fixtures/gemma4_speculative_prompts.json`:

```json
{
  "version": 1,
  "prompts": [
    {"id": "en_fact_128", "text": "<128-token English factual prompt>", "context_len": 128, "max_new_tokens": 32},
    {"id": "zh_fact_128", "text": "<128-token Chinese factual prompt>", "context_len": 128, "max_new_tokens": 32},
    {"id": "code_repeat_512", "text": "<512-token code prompt>", "context_len": 512, "max_new_tokens": 128},
    {"id": "en_explain_512", "text": "<512-token English explanation prompt>", "context_len": 512, "max_new_tokens": 128},
    {"id": "chat_en_2048", "text": "<2048-token English chat prompt>", "context_len": 2048, "max_new_tokens": 32},
    {"id": "chat_zh_2048", "text": "<2048-token Chinese chat prompt>", "context_len": 2048, "max_new_tokens": 32}
  ]
}
```

The placeholder `<...>` text must be replaced with actual long text, or the tool must deterministically expand a seed text to the target token length. The P1/P1.5 tool must assert `actual_prompt_tokens == context_len` and fail otherwise.

A simple deterministic expansion strategy (acceptable for this offline tool). The tool must keep and use the resulting token IDs directly; the expanded text is only for reporting.

```python
def _expand_to_token_ids(tokenizer: Any, seed: str, target_len: int) -> list[int]:
    tokens = tokenizer.encode(seed)
    if len(tokens) >= target_len:
        return tokens[:target_len]
    repeated = (seed + " ") * ((target_len // len(tokens)) + 2)
    tokens = tokenizer.encode(repeated)
    return tokens[:target_len]
```

- [ ] **Step 7: Run tests**

```bash
uv run pytest tests/test_gemma4_speculative_tokenizer_gate.py -q
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add tests/tools/gemma4_speculative_tokenizer_gate.py tests/test_gemma4_speculative_tokenizer_gate.py tests/tools/fixtures/gemma4_speculative_prompts.json
git commit -m "feat(speculative): harden tokenizer gate and add prompt fixture"
```

---

### Task 2: Fix P1 Offline Draft Prototype

**Files:**
- Modify: `tests/tools/gemma4_speculative_prototype.py`
- Modify: `tests/test_gemma4_speculative_prototype.py`

**Interfaces:**
- Consumes: shared tokenizer, two `LLM` instances, fixed prompt fixture.
- Produces: JSON with acceptance stats and decode-only baseline TPS.

- [ ] **Step 1: Run tokenizer gate before loading models**

In `main()`, before constructing `LLM`, call the gate function and exit if `passed` is false.

```python
from tests.tools.gemma4_speculative_tokenizer_gate import build_report, _load_tokenizer

target_tok = _load_tokenizer(args.target_model)
draft_tok = _load_tokenizer(args.draft_model)
prompts = _load_prompts(args.prompt_file)
gate = build_report(args.target_model, args.draft_model, target_tok, draft_tok, prompts)
if not gate["passed"]:
    print(json.dumps({"tokenizer_gate": gate}, indent=2), file=sys.stderr)
    return 2
```

- [ ] **Step 2: Add memory feasibility check with correct GPU measurement**

After both `LLM`s are loaded and warmed up, before speculative decode:

```python
import torch

peak_reserved_bytes = torch.cuda.max_memory_reserved()
free_bytes, total_bytes = torch.cuda.mem_get_info()
memory_ok = (free_bytes / total_bytes) >= 0.05
memory_report = {
    "peak_reserved_gb": peak_reserved_bytes / 1e9,
    "free_gb": free_bytes / 1e9,
    "total_gb": total_bytes / 1e9,
    "free_ratio": free_bytes / total_bytes,
    "memory_gate_passed": memory_ok,
}
```

Reset peak memory before loading each model. Constrain both instances to BS=1 and the smallest `max_model_len` that covers the largest fixture bucket plus output tokens plus draft K:

```python
max_context = max(p["context_len"] for p in prompts)
max_new_tokens = max(p["max_new_tokens"] for p in prompts)
max_k = 8
needed_model_len = max_context + max_new_tokens + max_k + 32
torch.cuda.reset_peak_memory_stats()
target_llm = LLM(
    model=args.target_model,
    max_model_len=needed_model_len,
    gpu_memory_utilization=args.gpu_memory_utilization,
    max_num_seqs=1,
    max_num_batched_tokens=1024,
)
torch.cuda.reset_peak_memory_stats()
draft_llm = LLM(
    model=args.draft_model,
    max_model_len=needed_model_len,
    gpu_memory_utilization=args.gpu_memory_utilization,
    max_num_seqs=1,
    max_num_batched_tokens=1024,
)
```

Wrap the second model load in a try/except that emits a structured memory-gate failure JSON on OOM.

- [ ] **Step 3: Pass token IDs directly to draft engine**

Replace text-based draft proposer with token-ID based generation. The draft model must consume the same prefix token IDs as the target.

Add a helper that injects a request with exact token IDs into an engine:

```python
def _add_request_with_token_ids(
    llm: LLM,
    request_id: str,
    input_ids: list[int],
    sampling_params: SamplingParams,
) -> None:
    """Enqueue a request using exact token IDs, bypassing text tokenization."""
    engine = llm.engine
    # The request_builder is lazily created on the first public add_request.
    if engine.request_builder is None:
        # Trigger initialization with a prompt that is immediately aborted.
        engine.add_request(
            "__init_request_builder__",
            "",
            SamplingParams(temperature=0.0, max_tokens=1),
        )
        engine.abort_request("__init_request_builder__")
    assert engine.request_builder is not None
    req = engine.request_builder.build(
        request_id=request_id,
        prompt="",
        sampling_params=sampling_params,
    )
    req.input_ids = list(input_ids)
    req.guarded_prompt = ""
    engine.scheduler.enqueue_request(request_id, req)
    admitted = engine.scheduler.admit_queued_requests(max_new=1)
    assert request_id in admitted, f"draft request {request_id} was not admitted"
```

Then the draft proposer becomes:

```python
def draft_proposer(prefix: list[int], generated: list[int], k: int) -> list[int]:
    context_ids = prefix + generated
    return generate_draft_tokens_from_ids(draft_llm, context_ids, k)
```

And `generate_draft_tokens_from_ids` uses `_add_request_with_token_ids`:

```python
def generate_draft_tokens_from_ids(draft_llm: LLM, context_ids: list[int], k: int) -> list[int]:
    sp = SamplingParams(temperature=0.0, max_tokens=k, min_tokens=1)
    req_id = f"draft_generate_{time.time_ns()}"
    _add_request_with_token_ids(draft_llm, req_id, context_ids, sp)

    final_output: Any | None = None
    step_count = 0
    step_budget = max(64, k * 4)
    try:
        while draft_llm.engine.active_request_count > 0 and step_count < step_budget:
            step_count += 1
            outs = draft_llm.engine.step()
            for out in outs:
                if out.request_id != req_id:
                    continue
                if out.finished or (
                    final_output is None and len(out.outputs[0].token_ids) >= k
                ):
                    final_output = out

        if final_output is None:
            raise RuntimeError(
                f"draft generation did not finish within step_budget={step_budget}; "
                f"active_request_count={draft_llm.engine.active_request_count}"
            )
    finally:
        if final_output is None:
            with suppress(Exception):
                draft_llm.engine.abort_request(req_id)

    generated = list(final_output.outputs[0].token_ids)
    # Sanity check: the engine consumed exactly the token IDs we provided.
    assert list(final_output.prompt_token_ids) == context_ids, (
        "draft engine prompt token IDs do not match input token IDs"
    )
    return generated[:k]
```

- [ ] **Step 4: Measure decode-only baseline TPS**

Split `baseline_greedy` timing into prefill and decode regions. Record the time of the first step that produces output tokens as the decode start.

```python
prefill_elapsed = first_decode_start - baseline_start
decode_elapsed = baseline_end - first_decode_start
# The first token is produced by the prefill step and is not part of decode TPS.
decode_tokens = max(0, len(baseline_token_ids) - 1)
baseline_decode_tps = decode_tokens / decode_elapsed if decode_elapsed > 0 else 0.0
```

Report:

```python
"baseline_prefill_time_s": prefill_elapsed,
"baseline_decode_time_s": decode_elapsed,
"baseline_decode_tps": baseline_decode_tps,
```

- [ ] **Step 5: Update speculative_decode to use token-ID draft input and per-prompt fixture**

Change `speculative_decode` signature to accept a `draft_proposer` that receives token IDs. The existing n-gram proposer and new draft-model proposer both conform to the same signature.

- [ ] **Step 6: Run mock tests**

```bash
uv run pytest tests/test_gemma4_speculative_prototype.py -q
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add tests/tools/gemma4_speculative_prototype.py tests/test_gemma4_speculative_prototype.py
git commit -m "feat(speculative): fix P1 draft prototype with tokenizer gate, memory gate, and token-ID draft input"
```

---

### Task 3: Add Verifier Mode to Gemma4 Forward

**Files:**
- Modify: `vllm/model_executor/models/gemma4/model.py`
- Create: `tests/test_gemma4_verifier_flag.py`

**Interfaces:**
- Consumes: `attn_metadata["verifier_return_all_logits"] == True`.
- Produces: logits of shape `(batch, seq_len, vocab_size)` when flag set; otherwise unchanged `(batch, 1, vocab_size)`.

- [ ] **Step 1: Modify `Gemma4ForConditionalGeneration.forward`**

```python
hidden = self.model(...)
return_all = bool(
    isinstance(attn_metadata, dict)
    and attn_metadata.get("verifier_return_all_logits", False)
)
hidden_slice = hidden if return_all else hidden[:, -1:, :]
if getattr(self.model.config, "tie_word_embeddings", False):
    logits = torch.nn.functional.linear(
        hidden_slice, self.model.embed_tokens.weight
    )
else:
    logits = self.lm_head(hidden_slice, lora_mapping)
```

- [ ] **Step 2: Add unit test for verifier flag with fake lm_head that accepts lora_mapping**

Create `tests/test_gemma4_verifier_flag.py`:

```python
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.models.gemma4.model import Gemma4ForConditionalGeneration


class FakeLMHead:
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        self.weight = torch.randn(vocab_size, hidden_size)

    def __call__(self, hidden: torch.Tensor, lora_mapping: Any) -> torch.Tensor:
        return torch.nn.functional.linear(hidden, self.weight)


def _make_model(hidden_size: int, vocab_size: int, tie: bool) -> Any:
    model = object.__new__(Gemma4ForConditionalGeneration)
    nn.Module.__init__(model)
    inner = torch.nn.Module()
    inner.config = SimpleNamespace(tie_word_embeddings=tie)
    inner.forward = lambda *args, **kwargs: torch.randn(1, 5, hidden_size)
    model.model = inner
    if tie:
        inner.embed_tokens = SimpleNamespace(
            weight=torch.randn(vocab_size, hidden_size)
        )
        model.lm_head = None
    else:
        model.lm_head = FakeLMHead(vocab_size, hidden_size)
    return model


def test_verifier_flag_returns_all_logits():
    model = _make_model(hidden_size=16, vocab_size=8, tie=False)
    attn_metadata = {"verifier_return_all_logits": True}
    logits = Gemma4ForConditionalGeneration.forward(
        model,
        torch.tensor([[1, 2, 3, 4, 5]]),
        torch.tensor([[0, 1, 2, 3, 4]]),
        [],
        attn_metadata,
    )
    assert tuple(logits.shape) == (1, 5, 8)


def test_verifier_flag_false_returns_last_logits():
    model = _make_model(hidden_size=16, vocab_size=8, tie=False)
    attn_metadata = {"verifier_return_all_logits": False}
    logits = Gemma4ForConditionalGeneration.forward(
        model,
        torch.tensor([[1, 2, 3, 4, 5]]),
        torch.tensor([[0, 1, 2, 3, 4]]),
        [],
        attn_metadata,
    )
    assert tuple(logits.shape) == (1, 1, 8)


def test_verifier_flag_tied_embeddings():
    model = _make_model(hidden_size=16, vocab_size=8, tie=True)
    attn_metadata = {"verifier_return_all_logits": True}
    logits = Gemma4ForConditionalGeneration.forward(
        model,
        torch.tensor([[1, 2, 3, 4, 5]]),
        torch.tensor([[0, 1, 2, 3, 4]]),
        [],
        attn_metadata,
    )
    assert tuple(logits.shape) == (1, 5, 8)
```

- [ ] **Step 3: Run test**

```bash
uv run pytest tests/test_gemma4_verifier_flag.py -q
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add vllm/model_executor/models/gemma4/model.py tests/test_gemma4_verifier_flag.py
git commit -m "feat(speculative): add verifier_return_all_logits metadata path"
```

---

### Task 4: Build Cached Verifier Runner

**Files:**
- Create: `tests/tools/gemma4_speculative_verifier_microbench.py`
- Create: `tests/test_gemma4_speculative_verifier_microbench.py`

**Interfaces:**
- Consumes: `target_llm`, existing request id / slot idx, prefix length, `input_ids` of shape `(1, K+1)`.
- Produces: logits `(1, K+1, vocab_size)`.

- [ ] **Step 1: Add helper `build_verifier_metadata`**

Use `InputBatchBuilder.build_prefill` as the reference. Build metadata for a single request with cached prefix length `C` and new input length `K+1`. Call `ensure_blocks` first.

```python
def build_verifier_metadata(
    kv_block_manager: Any,
    inf_config: Any,
    slot_idx: int,
    request_id: str,
    prefix_len: int,
    input_ids: torch.Tensor,
    num_layers: int,
) -> dict[str, Any]:
    device = input_ids.device
    bsz, seqlen = input_ids.shape
    total_len = prefix_len + seqlen

    # Grow the block table if the verifier will write past the current end.
    kv_block_manager.ensure_blocks(request_id, total_len)
    kv_block_manager.update_block_table_row(slot_idx, request_id)

    positions = torch.arange(
        prefix_len, total_len, device=device, dtype=torch.long
    ).unsqueeze(0)
    seq_lens = torch.tensor([total_len], device=device, dtype=torch.int32)
    block_table = kv_block_manager.block_table_for_slot(slot_idx).unsqueeze(0)
    query_start_loc = torch.tensor([0, seqlen], device=device, dtype=torch.int32)
    slot_mapping = torch.empty(seqlen, device=device, dtype=torch.long)
    compute_slot_mapping(
        query_start_loc,
        positions.view(-1),
        block_table,
        kv_block_manager.block_size,
        slot_mapping,
        pad_id=-1,
    )
    return {
        "slot_mapping": slot_mapping,
        "seq_lens": seq_lens,
        "seq_lens_cpu": [int(total_len)],
        "max_seq_len_cpu": int(total_len),
        "kv_start_indices": torch.tensor([prefix_len], device=device, dtype=torch.int32),
        "kv_start_indices_cpu": [prefix_len],
        "block_tables": block_table,
        "is_prefill": True,
        "verifier_return_all_logits": True,
        "kv_scale_cache": kv_block_manager.kv_scale_caches,
        "kv_cache_dtype": inf_config.kv_type,
        "k_scale": inf_config.k_scale,
        "v_scale": inf_config.v_scale,
        "config": inf_config,
        "lora_mapping": LoRAMapping.from_ids([None]),
        "linear_attn_carry": [None] * num_layers,
        "linear_conv_carry": [None] * num_layers,
    }
```

- [ ] **Step 2: Add `run_cached_verifier`**

```python
def run_cached_verifier(
    target_llm: LLM,
    request_id: str,
    prefix_len: int,
    input_ids: torch.Tensor,  # (1, K+1)
) -> torch.Tensor:
    engine = target_llm.engine
    kvbm = engine.kv_block_manager
    req = engine.scheduler.get_request(request_id)
    slot_idx = int(req.slot_idx)
    attn_metadata = build_verifier_metadata(
        kvbm,
        engine.inf_config,
        slot_idx,
        request_id,
        prefix_len,
        input_ids,
        len(engine.model.model.layers),
    )
    positions = torch.arange(
        prefix_len, prefix_len + input_ids.shape[1],
        device=input_ids.device, dtype=torch.long,
    ).unsqueeze(0)
    with torch.inference_mode():
        logits = engine.model(
            input_ids,
            positions,
            kvbm.kv_caches,
            attn_metadata,
        )
    return logits
```

- [ ] **Step 3: Add unit test with fake engine**

Create `tests/test_gemma4_speculative_verifier_microbench.py` with a mock engine and assert `build_verifier_metadata` sets `is_prefill=True`, `verifier_return_all_logits=True`, and correct positions.

```python
def test_build_verifier_metadata():
    mod = _load_module()
    inf_config = SimpleNamespace(kv_type="fp8", k_scale=1.0, v_scale=1.0)
    kvbm = _make_fake_kv_block_manager()
    input_ids = torch.tensor([[10, 20, 30]])
    meta = mod.build_verifier_metadata(
        kvbm,
        inf_config,
        slot_idx=0,
        request_id="r0",
        prefix_len=5,
        input_ids=input_ids,
        num_layers=2,
    )
    assert meta["is_prefill"] is True
    assert meta["verifier_return_all_logits"] is True
    assert torch.equal(meta["seq_lens"], torch.tensor([8], dtype=torch.int32))
    assert torch.equal(
        meta["kv_start_indices"], torch.tensor([5], dtype=torch.int32)
    )
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_gemma4_speculative_verifier_microbench.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add tests/tools/gemma4_speculative_verifier_microbench.py tests/test_gemma4_speculative_verifier_microbench.py
git commit -m "feat(speculative): add cached verifier runner for P1.5 microbench"
```

---

### Task 5: Implement P1.5 Multi-Step Microbench

**Files:**
- Modify: `tests/tools/gemma4_speculative_verifier_microbench.py`
- Modify: `tests/test_gemma4_speculative_verifier_microbench.py`

**Interfaces:**
- Consumes: target LLM, draft LLM, prompt fixture, K values.
- Produces: JSON with per-prompt/per-K results and aggregate gate decision.

- [ ] **Step 1: Construct target and draft LLM instances with BS=1 KV budgets**

In the tool's `main()`, compute the minimum required `max_model_len` from the fixture and create both instances with `max_num_seqs=1`:

```python
max_context = max(p["context_len"] for p in prompts)
max_new_tokens = max(p["max_new_tokens"] for p in prompts)
max_k = max(args.k_values)
needed_model_len = max_context + max_new_tokens + max_k + 32

target_llm = LLM(
    model=args.target_model,
    max_model_len=needed_model_len,
    gpu_memory_utilization=args.gpu_memory_utilization,
    max_num_seqs=1,
    max_num_batched_tokens=1024,
)
draft_llm = LLM(
    model=args.draft_model,
    max_model_len=needed_model_len,
    gpu_memory_utilization=args.gpu_memory_utilization,
    max_num_seqs=1,
    max_num_batched_tokens=1024,
)
```

- [ ] **Step 2: Implement dual-cache state machine**

Represent state as:

```python
@dataclass
class SpecState:
    target_seq_len: int
    draft_seq_len: int
    generated: list[int]
    last_emitted: int
    target_forwards: int = 0
    draft_forwards: int = 0
    catchup_forwards: int = 0
```

Invariant: at the start of every speculative step, both caches represent the same prefix and have identical `seq_len`; `last_emitted` is the most recently generated token and is not yet cached.

For each step:

```python
# 1. Draft generates K tokens from [prefix, last_emitted].
# 2. Verifier runs on [last_emitted, d_1, ..., d_K].
# 3. Accept/reject loop produces committed tokens.
# 4. Update target_seq_len and draft_seq_len per the dual-cache commit table.
# 5. If all drafts accepted, run one draft catch-up forward with d_K, then bonus becomes last_emitted.
```

Correct state table (C = cached seq_len at step start, y = last_emitted):

| State | Target seq_len / KV | Draft seq_len / KV | Next input |
|---|---|---|---|
| First reject d_1 | C+1: prefix, y | C+1: prefix, y | recovered |
| Reject d_i | C+i: prefix, y, d_1..d_(i-1) | C+i: same | recovered |
| All accepted, catch-up before | C+K+1: prefix, y, d_1..d_K | C+K: prefix, y, d_1..d_(K-1) | d_K catch-up |
| All accepted, catch-up after | C+K+1: prefix, y, d_1..d_K | C+K+1: same | bonus |

Recovered/bonus tokens are the next `last_emitted` and are not cached until the following step.

- [ ] **Step 3: Implement tool-only block-table truncate helper**

Because `KVBlockManager` has no safe snapshot API, each repetition starts from a fresh prefill. Within a repetition, use this helper to truncate after a partial reject:

```python
def _truncate_request_blocks(
    kv_block_manager: Any,
    request_id: str,
    new_seq_len: int,
) -> None:
    """Free blocks beyond new_seq_len and refresh the block-table row."""
    block_size = kv_block_manager.block_size
    needed_blocks = max(1, (new_seq_len + block_size - 1) // block_size)
    block_ids = kv_block_manager._request_blocks.get(request_id, [])
    if len(block_ids) > needed_blocks:
        tail_ids = block_ids[needed_blocks:]
        block_ids[needed_blocks:] = []
        kv_block_manager._allocator.free(tail_ids)
    slot_idx = kv_block_manager._request_slot_id.get(request_id)
    if slot_idx is not None:
        kv_block_manager.update_block_table_row(slot_idx, request_id)
```

- [ ] **Step 4: Implement cached draft K-token runner**

Add a helper that drives K greedy decode steps on a persistent draft request, reusing its KV cache. It does not use `SamplingParams.max_tokens` or the engine's finish logic; EOS is ignored so the draft request never finishes.

```python
def run_cached_draft_k(
    draft_llm: LLM,
    request_id: str,
    prefix_len: int,
    first_input: int,
    k: int,
) -> tuple[list[int], int]:
    """Run K greedy decode steps on the draft model using cached KV.

    Returns (draft_token_ids, final_draft_seq_len).
    """
    engine = draft_llm.engine
    kvbm = engine.kv_block_manager
    req = engine.scheduler.get_request(request_id)
    slot_idx = int(req.slot_idx)
    num_layers = len(engine.model.model.layers)
    device = engine.device

    draft_tokens: list[int] = []
    current_input = first_input
    current_seq_len = prefix_len

    for _ in range(k):
        kvbm.ensure_blocks(request_id, current_seq_len + 1)
        kvbm.update_block_table_row(slot_idx, request_id)

        input_ids = torch.tensor([[current_input]], device=device, dtype=torch.long)
        positions = torch.tensor([[current_seq_len]], device=device, dtype=torch.long)
        seq_lens = torch.tensor([current_seq_len + 1], device=device, dtype=torch.int32)
        block_table = kvbm.block_table_for_slot(slot_idx).unsqueeze(0)
        query_start_loc = torch.tensor([0, 1], device=device, dtype=torch.int32)
        slot_mapping = torch.empty(1, device=device, dtype=torch.long)
        compute_slot_mapping(
            query_start_loc,
            positions.view(-1),
            block_table,
            kvbm.block_size,
            slot_mapping,
            pad_id=-1,
        )

        attn_metadata = {
            "slot_mapping": slot_mapping,
            "seq_lens": seq_lens,
            "seq_lens_cpu": [int(current_seq_len + 1)],
            "max_seq_len_cpu": int(current_seq_len + 1),
            "positions_cpu": [int(current_seq_len)],
            "block_tables": block_table,
            "is_prefill": False,
            "kv_start_indices": positions.to(torch.int32),
            "kv_start_indices_cpu": [int(current_seq_len)],
            "linear_attn_carry": [None] * num_layers,
            "linear_conv_carry": [None] * num_layers,
            "kv_scale_cache": kvbm.kv_scale_caches,
            "kv_cache_dtype": engine.inf_config.kv_type,
            "k_scale": engine.inf_config.k_scale,
            "v_scale": engine.inf_config.v_scale,
            "config": engine.inf_config,
            "lora_mapping": LoRAMapping.from_ids([None]),
        }

        with torch.inference_mode():
            logits = engine.model(input_ids, positions, kvbm.kv_caches, attn_metadata)

        next_token = int(torch.argmax(logits[0, -1]).item())
        draft_tokens.append(next_token)
        current_input = next_token
        current_seq_len += 1

        # Keep the scheduler request in sync with the KV cache.
        req.seq_len = current_seq_len
        req.generated_ids.append(next_token)

    return draft_tokens, current_seq_len
```

- [ ] **Step 5: Implement multi-step measurement loop**

For each prompt and K:

```python
def _prefill_target(llm: LLM, prompt_ids: list[int]) -> tuple[str, int]:
    """Prefill target and return (request_id, first_output_token).

    Stops as soon as the first output token appears; the request remains alive
    for subsequent verifier forwards. After prefill, the cached length is reset
    to the prompt length and the first output token is returned as the initial
    last_emitted.
    """
    req_id = f"target_prefill_{time.time_ns()}"
    _add_request_with_token_ids(
        llm,
        req_id,
        prompt_ids,
        SamplingParams(temperature=0.0, max_tokens=999999),
    )
    first_token: int | None = None
    step_budget = max(64, len(prompt_ids) * 2)
    steps = 0
    while first_token is None and steps < step_budget:
        steps += 1
        outs = llm.engine.step()
        for out in outs:
            if out.request_id != req_id:
                continue
            tokens = list(out.outputs[0].token_ids)
            if tokens:
                first_token = int(tokens[0])
                break
    if first_token is None:
        raise RuntimeError(f"target prefill for {req_id} produced no output token")

    req = llm.engine.scheduler.get_request(req_id)
    req.seq_len = len(prompt_ids)
    req.generated_ids = []
    req.is_prefill = False
    return req_id, first_token


def _prefill_draft_persistent(llm: LLM, prompt_ids: list[int]) -> str:
    """Prefill draft and keep request alive for manual K-step decode."""
    req_id = f"draft_prefill_{time.time_ns()}"
    _add_request_with_token_ids(
        llm,
        req_id,
        prompt_ids,
        SamplingParams(temperature=0.0, max_tokens=999999),
    )
    saw_output = False
    step_budget = max(64, len(prompt_ids) * 2)
    steps = 0
    while not saw_output and steps < step_budget:
        steps += 1
        outs = llm.engine.step()
        for out in outs:
            if out.request_id != req_id:
                continue
            tokens = list(out.outputs[0].token_ids)
            if tokens:
                saw_output = True
                break

    if not saw_output:
        raise RuntimeError(f"draft prefill for {req_id} produced no output token")

    # Establish the invariant: cached length is exactly the prompt length,
    # and no generated tokens have been consumed yet.
    req = llm.engine.scheduler.get_request(req_id)
    req.seq_len = len(prompt_ids)
    req.generated_ids = []
    req.is_prefill = False
    return req_id


repetition_results: list[dict[str, Any]] = []
for rep in range(num_repetitions):
    target_req_id: str | None = None
    draft_req_id: str | None = None
    try:
        # Start from a fresh prefill for both models; prefill time is NOT measured.
        target_req_id, target_y = _prefill_target(target_llm, prompt_ids)
        draft_req_id = _prefill_draft_persistent(draft_llm, prompt_ids)
        target_snapshot_len = len(prompt_ids)
        draft_snapshot_len = len(prompt_ids)

        state = SpecState(
            target_seq_len=target_snapshot_len,
            draft_seq_len=draft_snapshot_len,
            generated=[],
            last_emitted=target_y,
        )

        torch.cuda.synchronize()
        start = time.perf_counter()
        while len(state.generated) < tokens_per_repetition:
            remaining = tokens_per_repetition - len(state.generated)
            effective_k = min(K, remaining - 1) if remaining > 1 else 0
            step_state_machine(
                target_llm, draft_llm, state, effective_k,
                target_req_id, draft_req_id,
            )
            if state.last_emitted in eos_ids:
                break
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    finally:
        if target_req_id is not None:
            target_llm.engine.abort_request(target_req_id)
        if draft_req_id is not None:
            draft_llm.engine.abort_request(draft_req_id)

    actual_committed = len(state.generated)
    repetition_results.append({
        "committed_tokens": actual_committed,
        "elapsed_s": elapsed,
        "predicted_tps": actual_committed / elapsed if elapsed > 0 else 0.0,
        "target_forwards": state.target_forwards,
        "draft_forwards": state.draft_forwards,
        "catchup_forwards": state.catchup_forwards,
    })
```

The `step_state_machine` signature is:

```python
def step_state_machine(
    target_llm: LLM,
    draft_llm: LLM,
    state: SpecState,
    k: int,
    target_req_id: str,
    draft_req_id: str,
) -> None:
    ...
```

It must:
- If `k == 0` (only one continuation token remains):
  - Build verifier input `[last_emitted]` and call `run_cached_verifier` to get one logit; increment `state.target_forwards` by 1.
  - Take `argmax(logits[:, 0])` as the next token, append it to `state.generated`, and set it as the new `last_emitted`.
  - Advance `state.target_seq_len` by 1 and sync the target request (`req.seq_len = state.target_seq_len`, `req.generated_ids = []`).
  - Do not run the draft model, do not advance `state.draft_seq_len`, and do not run catch-up.
  - This is the terminating tail of the repetition; both requests are released immediately afterward by the outer `finally`.
- Otherwise:
  - Call `run_cached_draft_k(draft_llm, draft_req_id, state.draft_seq_len, state.last_emitted, k)` to produce `d_1..d_k`; increment `state.draft_forwards` by k.
  - Build verifier input `[last_emitted, d_1, ..., d_k]` and call `run_cached_verifier`; increment `state.target_forwards` by 1.
  - Run the accept/reject loop against logits positions `0..k-1`.
  - On all accepted and remaining budget, take the bonus token from logits position k.
  - Update `state.target_seq_len` to the committed length and sync the target scheduler request (`req.seq_len = state.target_seq_len`, `req.generated_ids = []`) after every verifier forward.
  - On partial reject:
    - Update `state.draft_seq_len` to the accepted prefix length.
    - Call `_truncate_request_blocks(target_kvbm, target_req_id, state.target_seq_len)` and `_truncate_request_blocks(draft_kvbm, draft_req_id, state.draft_seq_len)`.
    - Sync the draft request: `draft_req.seq_len = state.draft_seq_len`, `draft_req.generated_ids = []`.
  - On all accepted, call `run_cached_draft_k` once more with `d_k` as input (catch-up), discard its output token, update `state.draft_seq_len` without consuming it, sync the draft request (`draft_req.seq_len = state.draft_seq_len`, `draft_req.generated_ids = []`), and increment `state.catchup_forwards` by 1.

- [ ] **Step 6: Compute bit-exact and performance gates per prompt/K**

Run a baseline greedy target decode with `max_tokens = tokens_per_repetition + 1`. The baseline's first token is produced by prefill and corresponds to `target_y`; P1.5 only counts the remaining `tokens_per_repetition` continuation tokens in `state.generated` and in decode TPS.

```python
# state.generated contains tokens after target_y; baseline_token_ids[0] == target_y.
actual_committed = len(state.generated)
bit_exact = (
    state.generated
    == baseline_token_ids[1 : 1 + actual_committed]
)
predicted_tps_values = [r["predicted_tps"] for r in repetition_results]
aggregate_predicted_tps = statistics.median(predicted_tps_values)
speedup = aggregate_predicted_tps / baseline_decode_tps
per_prompt_passed = (
    bit_exact
    and speedup >= 1.2
    and aggregate_predicted_tps >= 13.0
    and aggregate_predicted_tps >= baseline_decode_tps
)
```

Do not compute speedup as a ratio of two medians; compute each repetition's TPS, take the median, then divide by the single measured baseline decode TPS.

- [ ] **Step 7: Aggregate across context buckets**

For a fixed K, collect per-prompt results from 128/512/2048 prompts. Aggregate and gate:

```python
aggregate_predicted_tps = statistics.median(
    [r.aggregate_predicted_tps for r in all_contexts]
)
aggregate_baseline_tps = statistics.median(
    [r.baseline_decode_tps for r in all_contexts]
)
aggregate_speedup = aggregate_predicted_tps / aggregate_baseline_tps
passed = (
    all(r.bit_exact for r in all_contexts)
    and aggregate_speedup >= 1.2
    and aggregate_predicted_tps >= 13.0
    and all(r.aggregate_predicted_tps >= r.baseline_decode_tps for r in all_contexts)
)
```

- [ ] **Step 8: Add mock state-machine and truncate helper tests**

Test first-token reject, partial accept, all-accepted+catch-up, and EOS paths using fake logits tensors and fake `run_cached_verifier` / `run_cached_draft_k`.

Also add a unit test for `_truncate_request_blocks` using a fake `KVBlockManager` that exposes `_request_blocks`, `_request_slot_id`, `_allocator`, and `block_size`:

```python
def test_truncate_request_blocks_frees_tail_blocks():
    mod = _load_module()
    kvbm = _make_fake_kv_block_manager()
    kvbm._request_blocks["r0"] = [1, 2, 3, 4]
    kvbm._request_slot_id["r0"] = 0
    kvbm.block_size = 16

    mod._truncate_request_blocks(kvbm, "r0", new_seq_len=20)

    # 20 tokens need ceil(20/16)=2 blocks; tail blocks 3,4 should be freed.
    assert kvbm._request_blocks["r0"] == [1, 2]
    assert kvbm._allocator._allocated_ids == {1, 2}
```

- [ ] **Step 9: Run tests**

```bash
uv run pytest tests/test_gemma4_speculative_verifier_microbench.py -q
```

Expected: pass.

- [ ] **Step 10: Commit**

```bash
git add tests/tools/gemma4_speculative_verifier_microbench.py tests/test_gemma4_speculative_verifier_microbench.py
git commit -m "feat(speculative): add P1.5 multi-step verifier microbench"
```

---

### Task 6: Run P1/P1.5 on E2B and E4B Draft Models

**Files:**
- None (execution only).
- Produces: `docs/superpowers/reports/2026-07-09-gemma4-speculative-p15-results.md`.

**Interfaces:**
- Consumes: existing tools.
- Produces: JSON reports and a go/no-go decision.

- [ ] **Step 1: Run tokenizer gate for E2B**

```bash
uv run python tests/tools/gemma4_speculative_tokenizer_gate.py \
  --target-model models/gemma-4-26B-A4B-it-AWQ-4bit \
  --draft-model models/gemma-4-E2B-it-AWQ-INT4 \
  --json-out /tmp/e2b_tokenizer_gate.json
```

Expected: exit 0, `passed: true`.

- [ ] **Step 2: Run P1 acceptance sweep for E2B**

```bash
uv run python tests/tools/gemma4_speculative_prototype.py \
  --target-model models/gemma-4-26B-A4B-it-AWQ-4bit \
  --draft-model models/gemma-4-E2B-it-AWQ-INT4 \
  --prompt-file tests/tools/fixtures/gemma4_speculative_prompts.json \
  --json-out /tmp/e2b_p1_sweep.json
```

- [ ] **Step 3: Run P1.5 microbench for E2B**

```bash
uv run python tests/tools/gemma4_speculative_verifier_microbench.py \
  --target-model models/gemma-4-26B-A4B-it-AWQ-4bit \
  --draft-model models/gemma-4-E2B-it-AWQ-INT4 \
  --prompt-file tests/tools/fixtures/gemma4_speculative_prompts.json \
  --tokens-per-repetition 32 \
  --json-out /tmp/e2b_p15_microbench.json
```

- [ ] **Step 4: Repeat Steps 1-3 for E4B if E2B fails or is close**

```bash
# E4B tokenizer gate example
uv run python tests/tools/gemma4_speculative_tokenizer_gate.py \
  --target-model models/gemma-4-26B-A4B-it-AWQ-4bit \
  --draft-model models/gemma-4-E4B-it \
  --json-out /tmp/e4b_tokenizer_gate.json
```

- [ ] **Step 5: Record decision**

Create `docs/superpowers/reports/2026-07-09-gemma4-speculative-p15-results.md` summarizing:

- Which draft model(s) were tested.
- Tokenizer gate result.
- Memory gate result.
- Acceptance distribution per K and context bucket.
- P1.5 predicted TPS, speedup, bit-exact result.
- Go/no-go decision for P2.

- [ ] **Step 6: Commit report**

```bash
git add docs/superpowers/reports/2026-07-09-gemma4-speculative-p15-results.md
git commit -m "docs(speculative): P1/P1.5 E2B/E4B measurement report"
```

---

## Spec Coverage Check

| Spec Section | Implementing Task |
|---|---|
| P0 n-gram prototype | Already exists; no new tasks. |
| P1.0 tokenizer hard gate (full vocab, added tokens, special tokens, direct encode) | Task 1. |
| P1.0 tokenizer diagnostic (chat-template token IDs) | Task 1. |
| P1.1 memory feasibility gate | Task 2. |
| P1.2 draft token-ID alignment | Task 2. |
| P1.3 acceptance sweep + fixture | Task 1 + Task 2. |
| P1.5 minimal verifier primitive | Task 3 + Task 4. |
| P1.5 initial prefill + first y | Task 5 state machine. |
| P1.5 multi-step measurement + KV truncate/catch-up | Task 5. |
| P1.5 bit-exact gate | Task 5. |
| P1.5 performance gate across contexts | Task 5. |
| P2 scope/eligibility/logits/KV commit | Documented as future plan; no implementation in this plan. |

## Placeholder Scan

- No "TBD", "TODO", or "implement later" remain.
- Every task ends with a test or execution step and a commit.
- Exact file paths are provided for every created/modified file.
- Type signatures and return shapes are explicit.
- No decode/re-encode fallback for token-ID injection or context expansion.
- No non-existent `KVBlockManager` snapshot API is referenced; truncate helper uses `_request_slot_id`.
- Both target and draft LLM instances are constrained to `max_num_seqs=1` with a context-appropriate `max_model_len`.
- Baseline decode TPS excludes the prefill-produced first token.
- P1.5 uses a persistent draft request with `run_cached_draft_k` and a persistent target request for `run_cached_verifier`.
- P1.5 prefill helpers stop at the first output token and raise on missing output; each repetition aborts both requests in a `finally` block, including prefill failures.
- Continuation budget uses `baseline max_tokens = tokens_per_repetition + 1`; P1.5 clamps `effective_k` to the remaining budget and has a dedicated `k == 0` single-token verifier branch.
- Request cleanup lives outside the measured region; elapsed time stops before `abort_request`.
- Bit-exact comparison aligns `state.generated` with `baseline_token_ids[1:]`, and partial-reject/catch-up both sync `draft_req.seq_len` and clear `draft_req.generated_ids`.
- Final aggregate gate is a conjunction of bit-exact, 1.2x speedup, 13.0 tok/s floor, and per-bucket non-regression.
- No uncommitted E2B/E4B model loading changes are included in speculative tasks.

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-07-09-gemma4-speculative-decoding.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints.

Which approach do you want?
