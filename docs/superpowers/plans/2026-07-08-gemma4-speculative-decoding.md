# Gemma4-26B/31B Speculative Decoding — P0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline n-gram speculative-decoding prototype tool in `tests/tools/` that proves the acceptance loop and measurement methodology without touching the engine.

**Architecture:** The tool runs a target Gemma4 LLM greedily to establish a baseline, then re-runs the same prompt with n-gram draft tokens verified by a tool-only target forward/logits helper. All work stays in `tests/tools/gemma4_speculative_prototype.py` and `tests/test_gemma4_speculative_prototype.py`; no `vllm/` runtime code changes.

**Tech Stack:** Python 3.12, `uv run`, PyTorch, vLLM lite engine, pytest, argparse, json.

## Global Constraints

- Python 3.12, `uv run`.
- No runtime `os.environ` reads in `vllm/`; the tool uses CLI arguments only.
- P0 code lives in `tests/tools/` as offline experiments.
- Greedy (`temperature=0`) only.
- No engine integration, no KV-cache sharing, no scheduler changes.
- No permanent diagnostic hooks in the hot path.
- Target model default: `models/gemma-4-26B-A4B-it-AWQ-4bit`.
- Do not touch `LiteEngine`, `InputBatchBuilder`, or the KV block manager.
- `run_target_logits` must be implemented inside `tests/tools/gemma4_speculative_prototype.py` only; do not add new runtime APIs in `vllm/`.
- P0 JSON output must use `projected_tps` only; real `effective_tps` belongs to P1/P2.

---

## File Structure

- `tests/tools/gemma4_speculative_prototype.py`
  - N-gram draft proposal (`propose_ngram`).
  - Tool-only target logits helper (`run_target_logits`) that bypasses `LLM.generate()`.
  - Greedy baseline runner (`baseline_greedy`).
  - Speculative decode loop (`speculative_decode`).
  - CLI `main()` and JSON output.
- `tests/test_gemma4_speculative_prototype.py`
  - pytest unit tests loaded via `importlib.util.spec_from_file_location`, following the pattern in `tests/test_gemma4_strict_audit_smoke.py`.
  - All model-heavy paths are mocked so the suite runs without a GPU.

---

## Task 1: N-gram Proposal Helper

**Files:**
- Create: `tests/tools/gemma4_speculative_prototype.py`
- Test: `tests/test_gemma4_speculative_prototype.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `propose_ngram(prefix_token_ids: list[int], generated_token_ids: list[int], k: int, ngram_min: int = 2, ngram_max: int = 4) -> list[int]`
  - Searches `prefix_token_ids + generated_token_ids` for the most recent earlier occurrence of the suffix of `generated_token_ids` (lengths `ngram_max` down to `ngram_min`).
  - Returns up to `k` tokens that followed that earlier occurrence.
  - Returns an empty list if no match is found.
  - Must not match the suffix occurrence at the very end of `generated_token_ids`.

- [ ] **Step 1: Write the failing test**

In `tests/test_gemma4_speculative_prototype.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import json
import sys

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]


def _load_module() -> Any:
    p = _ROOT / "tests" / "tools" / "gemma4_speculative_prototype.py"
    spec = importlib.util.spec_from_file_location("gemma4_speculative_prototype", p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def proto_mod() -> Any:
    return _load_module()


def test_propose_ngram_finds_repeated_suffix(proto_mod: Any) -> None:
    prefix = [1, 2, 3, 4, 5]
    generated = [2, 3]
    # The suffix [2, 3] appeared earlier starting at index 1 in [1,2,3,4,5,2,3].
    # The 3 tokens after that earlier occurrence are [4, 5, 2].
    assert proto_mod.propose_ngram(prefix, generated, k=3, ngram_min=2, ngram_max=4) == [4, 5, 2]


def test_propose_ngram_prefers_longer_needle(proto_mod: Any) -> None:
    prefix = [10, 20, 30, 40]
    generated = [10, 20, 30]
    # n=3 needle [10,20,30] matches at index 0; return [40, 10].
    assert proto_mod.propose_ngram(prefix, generated, k=2, ngram_min=2, ngram_max=3) == [40, 10]


def test_propose_ngram_returns_empty_when_no_match(proto_mod: Any) -> None:
    assert proto_mod.propose_ngram([1, 2, 3], [4, 5], k=5, ngram_min=2, ngram_max=4) == []


def test_propose_ngram_does_not_match_final_suffix(proto_mod: Any) -> None:
    prefix = [1, 2]
    generated = [3, 4]
    # The only [3,4] is the final suffix itself; no earlier occurrence exists.
    assert proto_mod.propose_ngram(prefix, generated, k=2, ngram_min=2, ngram_max=2) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gemma4_speculative_prototype.py::test_propose_ngram_finds_repeated_suffix -v`

Expected: `AttributeError: module 'gemma4_speculative_prototype' has no attribute 'propose_ngram'` (or `ModuleNotFoundError`).

- [ ] **Step 3: Write the minimal implementation**

In `tests/tools/gemma4_speculative_prototype.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations


def propose_ngram(
    prefix_token_ids: list[int],
    generated_token_ids: list[int],
    k: int,
    ngram_min: int = 2,
    ngram_max: int = 4,
) -> list[int]:
    """Propose up to k draft tokens by matching the recent generated suffix.

    Searches prefix_token_ids + generated_token_ids for the most recent earlier
    occurrence of the suffix of generated_token_ids.  Returns the tokens that
    followed that earlier occurrence, up to k.
    """
    full = prefix_token_ids + generated_token_ids
    for n in range(ngram_max, ngram_min - 1, -1):
        if len(generated_token_ids) < n:
            continue
        needle = tuple(generated_token_ids[-n:])
        # The final occurrence (the suffix itself) ends at len(full) - n.
        # Search before that so we do not match the needle against itself.
        for i in range(len(full) - n - 1, -1, -1):
            if tuple(full[i : i + n]) == needle:
                start = i + n
                return full[start : start + k]
    return []
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gemma4_speculative_prototype.py -v`

Expected: all 4 `test_propose_ngram_*` tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/tools/gemma4_speculative_prototype.py tests/test_gemma4_speculative_prototype.py
git commit -m "feat(tools): n-gram draft proposal for Gemma4 speculative prototype"
```

---

## Task 2: Tool-Only Target Logits Helper

**Files:**
- Modify: `tests/tools/gemma4_speculative_prototype.py`
- Test: `tests/test_gemma4_speculative_prototype.py`

**Interfaces:**
- Consumes: an `llm` instance with `.model.model`, `.model.lm_head`, `.model.embed_tokens`, `.engine.inf_config`, and `.model.config`.
- Produces: `run_target_logits(llm, input_ids: torch.Tensor) -> torch.Tensor` of shape `[B, L, vocab_size]`.
  - Bypasses `LLM.generate()` black box.
  - Runs the inner `Gemma4TextModel` on the full sequence with no KV cache (mathematically correct for greedy verification).
  - Applies tied embeddings and final logit softcapping exactly like the model's own forward.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gemma4_speculative_prototype.py`:

```python
import torch


def test_run_target_logits_tie_word_embeddings(proto_mod: Any) -> None:
    vocab_size = 7
    hidden_size = 4
    seq_len = 5
    fake_weight = torch.randn(vocab_size, hidden_size)
    fake_hidden = torch.randn(1, seq_len, hidden_size)

    inner = SimpleNamespace(
        config=SimpleNamespace(
            tie_word_embeddings=True,
            final_logit_softcapping=30.0,
        ),
        embed_tokens=SimpleNamespace(weight=fake_weight),
        layers=[None, None],
        forward=lambda *args, **kwargs: fake_hidden,
        parameters=lambda: iter([fake_weight]),
    )
    llm = SimpleNamespace(
        model=SimpleNamespace(model=inner),
        engine=SimpleNamespace(inf_config={"dummy": True}),
    )

    logits = proto_mod.run_target_logits(llm, torch.tensor([1, 2, 3, 4, 5]))

    assert logits.shape == (1, seq_len, vocab_size)
    assert logits.abs().max().item() <= 30.5


def test_run_target_logits_untied_lm_head(proto_mod: Any) -> None:
    vocab_size = 7
    hidden_size = 4
    seq_len = 3
    fake_hidden = torch.randn(1, seq_len, hidden_size)
    lm_logits = torch.randn(1, seq_len, vocab_size)

    lm_head = SimpleNamespace(__call__=lambda hidden, lora_mapping: lm_logits)
    inner = SimpleNamespace(
        config=SimpleNamespace(
            tie_word_embeddings=False,
            final_logit_softcapping=None,
        ),
        embed_tokens=SimpleNamespace(weight=torch.randn(vocab_size, hidden_size)),
        layers=[None],
        forward=lambda *args, **kwargs: fake_hidden,
        parameters=lambda: iter([torch.randn(1)]),
    )
    llm = SimpleNamespace(
        model=SimpleNamespace(model=inner, lm_head=lm_head),
        engine=SimpleNamespace(inf_config={"dummy": True}),
    )

    logits = proto_mod.run_target_logits(llm, torch.tensor([1, 2, 3]))

    assert logits.shape == (1, seq_len, vocab_size)
    assert torch.equal(logits, lm_logits)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gemma4_speculative_prototype.py::test_run_target_logits_tie_word_embeddings -v`

Expected: `AttributeError: module 'gemma4_speculative_prototype' has no attribute 'run_target_logits'`.

- [ ] **Step 3: Write the minimal implementation**

Append to `tests/tools/gemma4_speculative_prototype.py`:

```python
import torch


def run_target_logits(llm, input_ids: torch.Tensor) -> torch.Tensor:
    """Return per-position target logits for greedy speculative verification.

    This bypasses LLM.generate() and runs the inner model on the full sequence
    with no KV cache.  For greedy decoding this is mathematically equivalent to
    the engine's decode path, just slower because attention is recomputed from
    scratch each call.
    """
    inner = llm.model.model
    device = next(inner.parameters()).device

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device)

    batch_size, seq_len = input_ids.shape
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    kv_caches = [None] * len(inner.layers)
    attn_metadata = {
        "config": llm.engine.inf_config,
        "is_prefill": True,
        "positions_cpu": list(range(seq_len)),
        "max_seq_len_cpu": seq_len,
    }

    with torch.inference_mode():
        hidden = inner(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            lora_mapping=None,
        )

        if getattr(inner.config, "tie_word_embeddings", False):
            logits = torch.nn.functional.linear(hidden, inner.embed_tokens.weight)
        else:
            logits = llm.model.lm_head(hidden, lora_mapping=None)

        final_softcap = getattr(inner.config, "final_logit_softcapping", None)
        if final_softcap is not None and float(final_softcap) > 0:
            logits = torch.tanh(logits / float(final_softcap)) * float(final_softcap)

    return logits
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gemma4_speculative_prototype.py::test_run_target_logits_tie_word_embeddings tests/test_gemma4_speculative_prototype.py::test_run_target_logits_untied_lm_head -v`

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/tools/gemma4_speculative_prototype.py tests/test_gemma4_speculative_prototype.py
git commit -m "feat(tools): tool-only target logits helper for speculative verification"
```

---

## Task 3: Greedy Baseline and Speculative Decode Loop

**Files:**
- Modify: `tests/tools/gemma4_speculative_prototype.py`
- Test: `tests/test_gemma4_speculative_prototype.py`

**Interfaces:**
- Consumes:
  - `baseline_greedy(llm, prompt_text: str, max_new_tokens: int) -> tuple[list[int], list[int]]` returning `(prompt_token_ids, generated_token_ids)`.
  - `run_target_logits(llm, input_ids)` from Task 2.
  - A draft proposer callable with signature `(prefix_token_ids, generated_token_ids, k) -> list[int]`.
- Produces:
  - `speculative_decode(llm, draft_proposer, prompt_token_ids: list[int], max_new_tokens: int, num_draft_tokens: int) -> dict`
    - `token_ids`: generated token ids.
    - `baseline_token_ids`: passed through for `bit_exact` comparison.
    - `bit_exact`: bool.
    - `accepted_total`, `proposed_total`, `acceptance_rate`, `target_forwards`.
    - `baseline_tps`, `speculative_tps`, `projected_tps`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gemma4_speculative_prototype.py`:

```python
def _make_mock_llm(reference_generated: list[int]) -> Any:
    """Return a mock LLM whose generate() emits reference_generated."""
    completion = SimpleNamespace(token_ids=list(reference_generated), text="")
    output = SimpleNamespace(
        prompt_token_ids=[1, 2, 3],
        outputs=[completion],
    )

    def _generate(prompts: list[str], sampling_params: Any) -> list[Any]:
        return [output]

    return SimpleNamespace(generate=_generate)


def _mock_run_target_logits(reference: list[int], prompt_len: int):
    """Return logits whose argmax matches the reference at each generated position."""
    def _run(llm: Any, input_ids: torch.Tensor) -> torch.Tensor:
        ids = input_ids[0].tolist()
        vocab_size = max(reference + ids) + 10
        logits = torch.full((1, len(ids), vocab_size), -1e9)
        for pos in range(prompt_len, len(ids)):
            token = reference[pos - prompt_len]
            logits[0, pos - 1, token] = 1e6
        return logits
    return _run


def test_speculative_decode_bit_exact_when_drafts_match(proto_mod: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    reference = [10, 11, 12, 13]
    prompt = [1, 2, 3]
    mock_llm = _make_mock_llm(reference)
    monkeypatch.setattr(proto_mod, "run_target_logits", _mock_run_target_logits(reference, len(prompt)))

    def draft_proposer(prefix: list[int], generated: list[int], k: int) -> list[int]:
        # Perfect oracle: return the next reference tokens.
        start = len(generated)
        return reference[start : start + k]

    result = proto_mod.speculative_decode(
        mock_llm,
        draft_proposer,
        prompt_token_ids=prompt,
        max_new_tokens=len(reference),
        num_draft_tokens=2,
    )

    assert result["token_ids"] == reference
    assert result["bit_exact"] is True
    assert result["acceptance_rate"] == 1.0
    assert result["accepted_total"] <= len(reference)


def test_speculative_decode_recovers_on_mismatch(proto_mod: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    reference = [10, 11, 12]
    prompt = [1, 2, 3]
    mock_llm = _make_mock_llm(reference)
    monkeypatch.setattr(proto_mod, "run_target_logits", _mock_run_target_logits(reference, len(prompt)))

    calls: list[int] = []

    def draft_proposer(prefix: list[int], generated: list[int], k: int) -> list[int]:
        calls.append(len(generated))
        if len(generated) == 0:
            return [99, reference[1]]  # first draft token mismatches
        return reference[len(generated) : len(generated) + k]

    result = proto_mod.speculative_decode(
        mock_llm,
        draft_proposer,
        prompt_token_ids=prompt,
        max_new_tokens=len(reference),
        num_draft_tokens=2,
    )

    assert result["token_ids"] == reference
    assert result["bit_exact"] is True
    assert result["accepted_total"] < result["proposed_total"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gemma4_speculative_prototype.py::test_speculative_decode_bit_exact_when_drafts_match -v`

Expected: `AttributeError: module 'gemma4_speculative_prototype' has no attribute 'speculative_decode'`.

- [ ] **Step 3: Write the minimal implementation**

Append to `tests/tools/gemma4_speculative_prototype.py`:

```python
import time
from typing import Callable

from vllm import LLM
from vllm.sampling_params import SamplingParams


def baseline_greedy(llm: LLM, prompt_text: str, max_new_tokens: int) -> tuple[list[int], list[int]]:
    """Greedy baseline run. Returns (prompt_token_ids, generated_token_ids)."""
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outputs = llm.generate([prompt_text], sp)
    output = outputs[0]
    return list(output.prompt_token_ids), list(output.outputs[0].token_ids)


def speculative_decode(
    llm: LLM,
    draft_proposer: Callable[[list[int], list[int], int], list[int]],
    prompt_token_ids: list[int],
    max_new_tokens: int,
    num_draft_tokens: int,
) -> dict[str, Any]:
    """Offline speculative decode loop using n-gram (or any) draft tokens.

    Verifies draft tokens with run_target_logits and accepts them while they
    match the greedy target.  On the first mismatch the recovered target token
    is used instead.  If all draft tokens are accepted, one bonus target token
    is sampled from the logit past the last accepted draft.
    """
    prefix = list(prompt_token_ids)
    generated: list[int] = []
    accepted_total = 0
    proposed_total = 0
    target_forwards = 0

    start_time = time.perf_counter()
    while len(generated) < max_new_tokens:
        target_forwards += 1
        current = prefix + generated
        proposed = draft_proposer(prefix, generated, num_draft_tokens)
        full_input = current + proposed
        logits = run_target_logits(llm, torch.tensor([full_input], dtype=torch.long))

        accept_start = len(current)
        accepted: list[int] = []
        rejected = False
        for i, d in enumerate(proposed):
            pred = int(torch.argmax(logits[0, accept_start + i - 1]).item())
            if pred == d:
                accepted.append(d)
            else:
                accepted.append(pred)
                rejected = True
                break

        if proposed and not rejected:
            # All drafts accepted: take one bonus target token.
            bonus_pos = accept_start + len(proposed) - 1
            accepted.append(int(torch.argmax(logits[0, bonus_pos]).item()))
        elif not accepted:
            # No drafts proposed: take exactly one target token.
            accepted.append(int(torch.argmax(logits[0, accept_start - 1]).item()))

        generated.extend(accepted)
        accepted_total += sum(1 for tok, draft in zip(accepted[: len(proposed)], proposed) if tok == draft)
        proposed_total += len(proposed)

        if rejected:
            # The recovered token was already appended as the last accepted token.
            pass

        if len(generated) >= max_new_tokens:
            generated = generated[:max_new_tokens]
            break
    elapsed = time.perf_counter() - start_time

    acceptance_rate = accepted_total / proposed_total if proposed_total > 0 else 0.0
    speculative_tps = len(generated) / elapsed if elapsed > 0 else 0.0

    return {
        "token_ids": generated,
        "baseline_token_ids": [],  # filled by CLI caller
        "bit_exact": False,        # filled by CLI caller
        "accepted_total": accepted_total,
        "proposed_total": proposed_total,
        "acceptance_rate": acceptance_rate,
        "target_forwards": target_forwards,
        "baseline_tps": 0.0,       # filled by CLI caller
        "speculative_tps": speculative_tps,
        "projected_tps": 0.0,      # filled by CLI caller
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gemma4_speculative_prototype.py::test_speculative_decode_bit_exact_when_drafts_match tests/test_gemma4_speculative_prototype.py::test_speculative_decode_recovers_on_mismatch -v`

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/tools/gemma4_speculative_prototype.py tests/test_gemma4_speculative_prototype.py
git commit -m "feat(tools): greedy baseline and speculative decode loop"
```

---

## Task 4: CLI, JSON Output, and Help

**Files:**
- Modify: `tests/tools/gemma4_speculative_prototype.py`
- Test: `tests/test_gemma4_speculative_prototype.py`

**Interfaces:**
- Consumes: everything from Tasks 1-3.
- Produces: `main(argv: list[str] | None = None) -> int` and a JSON file when `--json-out` is provided.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gemma4_speculative_prototype.py`:

```python
def test_cli_help_exits_zero(proto_mod: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["gemma4_speculative_prototype.py", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        proto_mod.main()
    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert "--target-model" in captured.out
    assert "--num-draft-tokens" in captured.out


def test_cli_writes_json(proto_mod: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model_dir = tmp_path / "fake-model"
    model_dir.mkdir()
    json_out = tmp_path / "out.json"

    reference = [10, 11]
    prompt = [1, 2, 3]

    def _mock_baseline(*args, **kwargs):
        return prompt, list(reference)

    def _mock_speculate(*args, **kwargs):
        return {
            "token_ids": list(reference),
            "baseline_token_ids": list(reference),
            "bit_exact": True,
            "accepted_total": 2,
            "proposed_total": 2,
            "acceptance_rate": 1.0,
            "target_forwards": 1,
            "baseline_tps": 0.0,
            "speculative_tps": 0.0,
            "projected_tps": 0.0,
        }

    monkeypatch.setattr(proto_mod, "baseline_greedy", _mock_baseline)
    monkeypatch.setattr(proto_mod, "speculative_decode", _mock_speculate)
    monkeypatch.setattr(
        proto_mod, "LLM", lambda **kwargs: SimpleNamespace(shutdown=lambda: None)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gemma4_speculative_prototype.py",
            "--target-model",
            str(model_dir),
            "--prompt",
            "hello",
            "--max-new-tokens",
            "2",
            "--json-out",
            str(json_out),
        ],
    )

    rc = proto_mod.main()
    assert rc == 0
    assert json_out.exists()
    data = json.loads(json_out.read_text())
    assert data["bit_exact"] is True
    assert "acceptance_rate" in data
    assert "projected_tps" in data
```

Add `import json` and `import sys` at the top of the test file if not already present.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gemma4_speculative_prototype.py::test_cli_help_exits_zero -v`

Expected: `AttributeError: module 'gemma4_speculative_prototype' has no attribute 'main'`.

- [ ] **Step 3: Write the minimal implementation**

Append to `tests/tools/gemma4_speculative_prototype.py`:

```python
import argparse
import json
import os
import sys
from pathlib import Path


def _default_model_path() -> str:
    return "models/gemma-4-26B-A4B-it-AWQ-4bit"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline n-gram speculative decoding prototype for Gemma4.",
    )
    parser.add_argument("--target-model", default=_default_model_path())
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--num-draft-tokens", type=int, default=5)
    parser.add_argument("--ngram-min", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--json-out", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.prompt_file:
        prompt_text = Path(args.prompt_file).read_text()
    else:
        prompt_text = args.prompt

    target_model_path = args.target_model
    if not os.path.isdir(target_model_path):
        print(f"[ERROR] target model directory not found: {target_model_path}", file=sys.stderr)
        return 2

    target_llm = LLM(
        model=target_model_path,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=1024,
    )

    baseline_start = time.perf_counter()
    prompt_token_ids, baseline_token_ids = baseline_greedy(
        target_llm, prompt_text, args.max_new_tokens
    )
    baseline_elapsed = time.perf_counter() - baseline_start

    def draft_proposer(prefix: list[int], generated: list[int], k: int) -> list[int]:
        return propose_ngram(prefix, generated, k, args.ngram_min, args.ngram_max)

    spec_result = speculative_decode(
        target_llm,
        draft_proposer,
        prompt_token_ids,
        max_new_tokens=args.max_new_tokens,
        num_draft_tokens=args.num_draft_tokens,
    )

    baseline_tps = len(baseline_token_ids) / baseline_elapsed if baseline_elapsed > 0 else 0.0
    spec_result["baseline_token_ids"] = baseline_token_ids
    spec_result["bit_exact"] = spec_result["token_ids"] == baseline_token_ids
    spec_result["baseline_tps"] = baseline_tps

    target_forwards = spec_result["target_forwards"]
    if target_forwards > 0:
        theoretical_speedup = 1.0 + spec_result["accepted_total"] / target_forwards
    else:
        theoretical_speedup = 1.0
    spec_result["projected_tps"] = baseline_tps * theoretical_speedup

    output = {
        "prompt_text": prompt_text,
        "target_model": target_model_path,
        "num_draft_tokens": args.num_draft_tokens,
        "ngram_min": args.ngram_min,
        "ngram_max": args.ngram_max,
        "baseline_tokens": baseline_token_ids,
        "speculative_tokens": spec_result["token_ids"],
        "bit_exact": spec_result["bit_exact"],
        "accepted_total": spec_result["accepted_total"],
        "proposed_total": spec_result["proposed_total"],
        "acceptance_rate": spec_result["acceptance_rate"],
        "baseline_tps": spec_result["baseline_tps"],
        "speculative_tps": spec_result["speculative_tps"],
        "projected_tps": spec_result["projected_tps"],
    }

    print(json.dumps(output, indent=2, sort_keys=True))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    target_llm.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gemma4_speculative_prototype.py -v`

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/tools/gemma4_speculative_prototype.py tests/test_gemma4_speculative_prototype.py
git commit -m "feat(tools): CLI and JSON output for Gemma4 speculative prototype"
```

---

## Task 5: Manual Real-Model Smoke Run

**Files:** none (manual verification).

**Interfaces:**
- Consumes: the finished tool.
- Produces: a JSON artifact and a sanity check that the prototype runs against real weights.

- [ ] **Step 1: Run the tool against the 26B target**

Command:

```bash
uv run python tests/tools/gemma4_speculative_prototype.py \
  --target-model models/gemma-4-26B-A4B-it-AWQ-4bit \
  --prompt "The capital of France is" \
  --max-new-tokens 32 \
  --num-draft-tokens 5 \
  --json-out /tmp/gemma4_p0_ngram.json
```

Expected: exits 0, prints JSON, `bit_exact` is `true`, `acceptance_rate` is reported.

- [ ] **Step 2: Record the result**

Append a one-line note to the bottom of `/tmp/gemma4_p0_ngram.json` or to a local notes file, e.g.:

```text
P0 smoke: acceptance_rate=X.XX, bit_exact=true on 2026-07-08.
```

- [ ] **Step 3: Commit (no code change, optional)**

If the smoke run passes and the only changes are the files already committed, no extra commit is needed.

---

## Self-Review Checklist

**1. Spec coverage:**
- Offline n-gram prototype in `tests/tools/`: Task 4.
- No engine changes: only `tests/tools/` and `tests/test_*.py` are touched.
- Per-position target logits instead of `LLM.generate()`: Task 2.
- Greedy-only / temperature=0: Task 3 `SamplingParams` and verification argmax.
- Bit-exact output: Task 3 `bit_exact` comparison.
- JSON outputs including `acceptance_rate` and `projected_tps`: Task 4.
- P1 tokenizer gate is documented in the spec but not implemented here; this plan is P0 only.

**2. Placeholder scan:**
- No "TBD", "TODO", "implement later", "fill in details".
- No vague "add error handling" or "write tests for the above".
- Every code step contains the actual function/test code.
- Exact file paths are used everywhere.

**3. Type consistency:**
- `propose_ngram` signature matches its usage in the `draft_proposer` closure.
- `run_target_logits` accepts `torch.Tensor` and returns `torch.Tensor`.
- `speculative_decode` returns the dict keys the CLI and tests expect.
- `main` returns `int` and writes the same JSON it prints.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-07-08-gemma4-speculative-decoding.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints.

**Which approach?**
