# SamplingDriver Batch Vectorization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `SamplingDriver.sample_batch_tokens` into a vectorized `PenaltyEncoder` and a reusable `Sampler`, eliminating the per-request Python loops that dominate decode-step sampling latency.

**Architecture:** Move EOS helpers and per-row penalty logic into a new `vllm/engine/sampling/` package. `PenaltyEncoder` applies repetition / frequency / presence penalties and EOS / anti-template masks in batch using `torch.gather`/`scatter` with a dummy-column padding trick; rows that cannot be vectorized (structured output constraints, non-empty context bias) fall back to per-row processing. `Sampler` owns the already-vectorized temperature / top-k / top-p / multinomial logic. `SamplingDriver` becomes an orchestrator and retains an opt-out legacy path.

**Tech Stack:** Python 3.12, PyTorch 2.9+ (ROCm 7.2), `uv`, `pytest`, `ruff`.

## Global Constraints

- All tensor logic uses PyTorch; no NumPy in the hot path.
- New files follow the repository's `SPDX-License-Identifier: Apache-2.0` header.
- Every new public function/class has type hints and a docstring.
- Tests must include numerical parity against the existing per-row implementation.
- The public API of `SamplingDriver` (`sample_next_token`, `sample_batch_tokens`, `completion_eos_ids`) must remain unchanged.
- Opt-out environment variable `FASTINFERENCE_USE_LEGACY_SAMPLING=1` restores the old per-row path for one release.

---

## File Structure

| File | Responsibility |
|---|---|
| `vllm/engine/sampling/__init__.py` | Package marker and re-exports. |
| `vllm/engine/sampling/utils.py` | EOS / stop-token helper functions extracted from `sampling_driver.py`. |
| `vllm/engine/sampling/penalty_encoder.py` | `PenaltyEncoder`: batch-level penalty / bias / mask application with per-row fallback. |
| `vllm/engine/sampling/sampler.py` | `Sampler`: vectorized temperature / top-k / top-p / multinomial sampling. |
| `vllm/engine/sampling_driver.py` | Thin orchestrator; retains single-request fallback and legacy path. |
| `tests/test_sampling_utils.py` | Unit tests for EOS helper parity. |
| `tests/test_penalty_encoder.py` | Numerical parity tests for vectorized vs per-row penalties. |
| `tests/test_sampler.py` | Unit tests for greedy and multinomial batch sampling. |
| `benchmarks/sampling_driver_microbenchmark.py` | Reproducible microbenchmark used to validate the speed-up. |

---

## Task 1: Extract EOS helpers into `vllm/engine/sampling/utils.py`

**Files:**
- Create: `vllm/engine/sampling/__init__.py`
- Create: `vllm/engine/sampling/utils.py`
- Modify: `vllm/engine/sampling_driver.py:13-48`

**Interfaces:**
- Consumes: nothing new.
- Produces: `hf_config_eos_token_ids(hf_config)` and `eos_stop_token_ids_for_sampling(tokenizer, sp, hf_config)` with the same signatures as today.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sampling_utils.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock

from vllm.engine.sampling.utils import eos_stop_token_ids_for_sampling
from vllm.sampling_params import SamplingParams


def test_eos_stop_token_ids_deduplicates_sources() -> None:
    tokenizer = Mock()
    tokenizer.eos_token_id = [1, 2]
    sp = SamplingParams(stop_token_ids=[2, 3])
    assert eos_stop_token_ids_for_sampling(tokenizer, sp, None) == [1, 2, 3]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_sampling_utils.py -v
```

Expected: `ModuleNotFoundError: No module named 'vllm.engine.sampling'`

- [ ] **Step 3: Create the sampling package and move helpers**

Create `vllm/engine/sampling/__init__.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from vllm.engine.sampling.utils import (
    eos_stop_token_ids_for_sampling,
    hf_config_eos_token_ids,
)

__all__ = ["eos_stop_token_ids_for_sampling", "hf_config_eos_token_ids"]
```

Create `vllm/engine/sampling/utils.py` by copying the two helper functions from `vllm/engine/sampling_driver.py` lines 13-48.

- [ ] **Step 4: Update `sampling_driver.py` to import from the new package**

Replace the helper definitions in `vllm/engine/sampling_driver.py` with:

```python
from vllm.engine.sampling.utils import eos_stop_token_ids_for_sampling
```

Remove the now-duplicated `hf_config_eos_token_ids` function if it is no longer used locally.

- [ ] **Step 5: Run the tests**

```bash
uv run pytest tests/test_sampling_utils.py -v
```

Expected: `1 passed`

- [ ] **Step 6: Run lint on changed files**

```bash
uv run ruff check vllm/engine/sampling/ vllm/engine/sampling_driver.py tests/test_sampling_utils.py
```

Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add vllm/engine/sampling/ vllm/engine/sampling_driver.py tests/test_sampling_utils.py
git commit -m "refactor(sampling): extract EOS helpers into vllm.engine.sampling package"
```

---

## Task 2: Implement vectorized `PenaltyEncoder`

**Files:**
- Create: `vllm/engine/sampling/penalty_encoder.py`
- Modify: `vllm/engine/sampling/__init__.py`
- Create: `tests/test_penalty_encoder.py`

**Interfaces:**
- Consumes: `eos_stop_token_ids_for_sampling` from Task 1.
- Produces: `PenaltyEncoder.encode(logits: torch.Tensor, requests: list[RequestState]) -> torch.Tensor`; `PenaltyEncoder.encode_row(logits: torch.Tensor, request: RequestState) -> torch.Tensor`.

- [ ] **Step 1: Write the failing parity test**

Create `tests/test_penalty_encoder.py`:

```python
# SPDX-License-Identifier: Apache-2.0
import torch

from vllm.engine.request_state import RequestState
from vllm.engine.sampling.penalty_encoder import PenaltyEncoder
from vllm.policies.base import GenerationPolicies
from vllm.sampling_params import SamplingParams


class NoOpOutputProcessor:
    def apply_context_bias(self, logits, generated_ids, sampling_params, bias_token_ids, is_capital_question):
        return logits


def _make_driver():
    class FakeTokenizer:
        eos_token_id = 2
    policies = GenerationPolicies(backend=NoOpOutputProcessor())
    return PenaltyEncoder(FakeTokenizer(), None, policies)


def test_vectorized_matches_per_row_repetition_penalty() -> None:
    encoder = _make_driver()
    vocab_size = 128
    requests = []
    for i in range(4):
        sp = SamplingParams(temperature=0.7, repetition_penalty=1.0 + i * 0.1)
        req = RequestState(
            request_id=f"r{i}",
            prompt="",
            input_ids=[1, 2, 3],
            sampling_params=sp,
            generated_ids=[10, 20, 30, 10],
        )
        requests.append(req)
    logits = torch.randn(4, vocab_size)
    vectorized = encoder.encode(logits, requests)
    for i, req in enumerate(requests):
        expected = encoder.encode_row(logits[i], req)
        assert torch.allclose(vectorized[i], expected, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_penalty_encoder.py::test_vectorized_matches_per_row_repetition_penalty -v
```

Expected: `ModuleNotFoundError` or `ImportError` for `PenaltyEncoder`.

- [ ] **Step 3: Implement `PenaltyEncoder`**

Create `vllm/engine/sampling/penalty_encoder.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch

from vllm.engine.request_state import RequestState
from vllm.engine.sampling.utils import eos_stop_token_ids_for_sampling


class PenaltyEncoder:
    """Apply sampling penalties, biases and masks to a batch of logits."""

    def __init__(self, tokenizer: Any, hf_config: Any | None, policies: Any) -> None:
        self.tokenizer = tokenizer
        self.hf_config = hf_config
        self.policies = policies

    def encode(
        self,
        logits: torch.Tensor,
        requests: list[RequestState],
    ) -> torch.Tensor:
        """Return penalty-adjusted logits for the whole batch."""
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        batch_size, vocab_size = logits.shape
        device = logits.device
        adjusted = logits.float().clone()

        fallback_rows = self._find_fallback_rows(requests)
        vectorized_rows = [i for i in range(batch_size) if i not in fallback_rows]

        if vectorized_rows:
            self._apply_vectorized_penalties(adjusted, requests, vectorized_rows, vocab_size)

        for i in fallback_rows:
            adjusted[i] = self.encode_row(adjusted[i], requests[i])

        return adjusted

    def encode_row(
        self,
        logits: torch.Tensor,
        request: RequestState,
    ) -> torch.Tensor:
        """Per-row fallback that mirrors the legacy penalty logic."""
        row = logits.float().clone()
        sp = request.sampling_params
        generated_ids = request.generated_ids
        vocab_size = row.numel()

        # Context bias and structured constraints are inherently per-row.
        row = self.policies.apply_context_bias(
            row,
            generated_ids,
            sp,
            request.capital_question_bias_token_ids,
            request.is_chinese_capital_question,
        )
        if request.structured_output_constraint is not None:
            row = request.structured_output_constraint.apply(row, request)

        rp = float(getattr(sp, "repetition_penalty", 1.0) or 1.0)
        if rp > 1.0 and generated_ids:
            for tid in set(int(t) for t in generated_ids):
                if 0 <= tid < vocab_size:
                    if row[tid] > 0:
                        row[tid] /= rp
                    else:
                        row[tid] *= rp

        fp = float(getattr(sp, "frequency_penalty", 0.0) or 0.0)
        if abs(fp) > 1e-12 and generated_ids:
            from collections import Counter
            for tid, cnt in Counter(int(t) for t in generated_ids).items():
                if 0 <= tid < vocab_size:
                    row[tid] -= fp * float(cnt)

        pp = float(getattr(sp, "presence_penalty", 0.0) or 0.0)
        if abs(pp) > 1e-12 and generated_ids:
            for tid in set(int(t) for t in generated_ids):
                if 0 <= tid < vocab_size:
                    row[tid] -= pp

        if not getattr(sp, "ignore_eos", False):
            mt = int(getattr(sp, "min_tokens", 0) or 0)
            if len(generated_ids) < mt:
                for tid in eos_stop_token_ids_for_sampling(self.tokenizer, sp, self.hf_config):
                    if 0 <= tid < vocab_size:
                        row[tid] = float("-inf")
        else:
            for tid in eos_stop_token_ids_for_sampling(self.tokenizer, sp, self.hf_config):
                if 0 <= tid < vocab_size:
                    row[tid] = float("-inf")

        anti_template = request.anti_template_token_ids
        if anti_template and len(generated_ids) < 12:
            for tid in anti_template:
                if 0 <= tid < vocab_size:
                    row[tid] -= 60.0

        return row

    def _find_fallback_rows(self, requests: list[RequestState]) -> set[int]:
        fallback: set[int] = set()
        for i, req in enumerate(requests):
            if req.structured_output_constraint is not None:
                fallback.add(i)
            if req.capital_question_bias_token_ids or req.is_chinese_capital_question:
                fallback.add(i)
        return fallback

    def _apply_vectorized_penalties(
        self,
        logits: torch.Tensor,
        requests: list[RequestState],
        rows: list[int],
        vocab_size: int,
    ) -> None:
        device = logits.device
        row_indices = torch.tensor(rows, dtype=torch.long, device=device)
        row_logits = logits[row_indices]

        # --- repetition penalty (multiplicative) ---
        unique_lists = [list(set(int(t) for t in requests[i].generated_ids)) for i in rows]
        max_len = max((len(u) for u in unique_lists), default=0)
        if max_len:
            dummy = vocab_size
            idx = torch.full((len(rows), max_len), dummy, dtype=torch.long, device=device)
            mask = torch.zeros((len(rows), max_len), dtype=torch.bool, device=device)
            for r, uids in enumerate(unique_lists):
                if uids:
                    idx[r, : len(uids)] = torch.tensor(uids, dtype=torch.long, device=device)
                    mask[r, : len(uids)] = True

            padded = torch.cat([row_logits, torch.zeros(len(rows), 1, device=device)], dim=1)
            selected = torch.gather(padded, 1, idx)
            rps = torch.tensor(
                [float(getattr(requests[i].sampling_params, "repetition_penalty", 1.0) or 1.0) for i in rows],
                dtype=torch.float,
                device=device,
            ).unsqueeze(1)
            factors = torch.where(selected > 0, 1.0 / rps.clamp(min=1e-6), rps)
            factors = torch.where(mask, factors, torch.ones_like(factors))

            multipliers = torch.ones(len(rows), vocab_size + 1, dtype=torch.float, device=device)
            multipliers.scatter_(1, idx, factors)
            row_logits.mul_(multipliers[:, :vocab_size])

        # --- frequency & presence penalties (additive) ---
        # Frequency
        freq_lists = []
        freq_counts = []
        max_freq = 0
        for i in rows:
            from collections import Counter
            counts = Counter(int(t) for t in requests[i].generated_ids)
            tids = list(counts.keys())
            cnts = [counts[t] for t in tids]
            freq_lists.append(tids)
            freq_counts.append(cnts)
            max_freq = max(max_freq, len(tids))

        if max_freq:
            dummy = vocab_size
            idx = torch.full((len(rows), max_freq), dummy, dtype=torch.long, device=device)
            vals = torch.zeros((len(rows), max_freq), dtype=torch.float, device=device)
            fps = torch.tensor(
                [float(getattr(requests[i].sampling_params, "frequency_penalty", 0.0) or 0.0) for i in rows],
                dtype=torch.float,
                device=device,
            ).unsqueeze(1)
            for r, (tids, cnts) in enumerate(zip(freq_lists, freq_counts)):
                if tids:
                    idx[r, : len(tids)] = torch.tensor(tids, dtype=torch.long, device=device)
                    vals[r, : len(tids)] = -fps[r].item() * torch.tensor(cnts, dtype=torch.float, device=device)
            additive = torch.zeros(len(rows), vocab_size + 1, dtype=torch.float, device=device)
            additive.scatter_add_(1, idx, vals)
            row_logits.add_(additive[:, :vocab_size])

        # Presence
        pres_lists = [list(set(int(t) for t in requests[i].generated_ids)) for i in rows]
        max_pres = max((len(u) for u in pres_lists), default=0)
        if max_pres:
            dummy = vocab_size
            idx = torch.full((len(rows), max_pres), dummy, dtype=torch.long, device=device)
            pps = torch.tensor(
                [float(getattr(requests[i].sampling_params, "presence_penalty", 0.0) or 0.0) for i in rows],
                dtype=torch.float,
                device=device,
            ).unsqueeze(1)
            vals = -pps.expand(-1, max_pres)
            mask = torch.zeros((len(rows), max_pres), dtype=torch.bool, device=device)
            for r, uids in enumerate(pres_lists):
                if uids:
                    idx[r, : len(uids)] = torch.tensor(uids, dtype=torch.long, device=device)
                    mask[r, : len(uids)] = True
            vals = torch.where(mask, vals, torch.zeros_like(vals))
            additive = torch.zeros(len(rows), vocab_size + 1, dtype=torch.float, device=device)
            additive.scatter_add_(1, idx, vals)
            row_logits.add_(additive[:, :vocab_size])

        # --- EOS mask ---
        eos_lists = [eos_stop_token_ids_for_sampling(self.tokenizer, requests[i].sampling_params, self.hf_config) for i in rows]
        first = eos_lists[0]
        uniform = all(e == first for e in eos_lists)
        if uniform and first:
            tids = [tid for tid in first if 0 <= tid < vocab_size]
            if tids:
                mask_positions = torch.tensor(tids, dtype=torch.long, device=device)
                for r, i in enumerate(rows):
                    sp = requests[i].sampling_params
                    ignore_eos = getattr(sp, "ignore_eos", False)
                    mt = int(getattr(sp, "min_tokens", 0) or 0)
                    if ignore_eos or len(requests[i].generated_ids) < mt:
                        row_logits[r, mask_positions] = float("-inf")
        else:
            for r, i in enumerate(rows):
                for tid in eos_lists[r]:
                    if 0 <= tid < vocab_size:
                        sp = requests[i].sampling_params
                        ignore_eos = getattr(sp, "ignore_eos", False)
                        mt = int(getattr(sp, "min_tokens", 0) or 0)
                        if ignore_eos or len(requests[i].generated_ids) < mt:
                            row_logits[r, tid] = float("-inf")

        # --- anti-template mask ---
        for r, i in enumerate(rows):
            anti = requests[i].anti_template_token_ids
            if anti and len(requests[i].generated_ids) < 12:
                for tid in anti:
                    if 0 <= tid < vocab_size:
                        row_logits[r, tid] -= 60.0

        logits[row_indices] = row_logits
```

- [ ] **Step 4: Re-export `PenaltyEncoder` from the package**

Update `vllm/engine/sampling/__init__.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from vllm.engine.sampling.penalty_encoder import PenaltyEncoder
from vllm.engine.sampling.utils import (
    eos_stop_token_ids_for_sampling,
    hf_config_eos_token_ids,
)

__all__ = [
    "eos_stop_token_ids_for_sampling",
    "hf_config_eos_token_ids",
    "PenaltyEncoder",
]
```

- [ ] **Step 5: Run the parity test**

```bash
uv run pytest tests/test_penalty_encoder.py::test_vectorized_matches_per_row_repetition_penalty -v
```

Expected: `1 passed`

- [ ] **Step 6: Add remaining parity tests**

Add tests for frequency penalty, presence penalty, EOS masking with `min_tokens` / `ignore_eos`, anti-template masking, and mixed batches with fallback rows. Each test asserts `torch.allclose(encoder.encode(logits, requests)[i], encoder.encode_row(logits[i], requests[i]))` for every row.

Run:

```bash
uv run pytest tests/test_penalty_encoder.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Lint changed files**

```bash
uv run ruff check vllm/engine/sampling/ tests/test_penalty_encoder.py
```

Expected: `All checks passed!`

- [ ] **Step 8: Commit**

```bash
git add vllm/engine/sampling/ tests/test_penalty_encoder.py
git commit -m "feat(sampling): vectorized PenaltyEncoder with per-row fallback"
```

---

## Task 3: Extract `Sampler` from `SamplingDriver`

**Files:**
- Create: `vllm/engine/sampling/sampler.py`
- Modify: `vllm/engine/sampling/__init__.py`
- Create: `tests/test_sampler.py`

**Interfaces:**
- Consumes: nothing new (self-contained).
- Produces: `Sampler.sample(logits: torch.Tensor, requests: list[RequestState]) -> list[int]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sampler.py`:

```python
# SPDX-License-Identifier: Apache-2.0
import torch

from vllm.engine.request_state import RequestState
from vllm.engine.sampling.sampler import Sampler
from vllm.sampling_params import SamplingParams


def test_sampler_greedy_batch() -> None:
    sp = SamplingParams(temperature=0.0)
    requests = [
        RequestState(request_id="r0", prompt="", input_ids=[1], sampling_params=sp),
        RequestState(request_id="r1", prompt="", input_ids=[1], sampling_params=sp),
    ]
    logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    sampler = Sampler()
    tokens = sampler.sample(logits, requests)
    assert tokens == [2, 0]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_sampler.py::test_sampler_greedy_batch -v
```

Expected: `ModuleNotFoundError` for `Sampler`.

- [ ] **Step 3: Implement `Sampler`**

Create `vllm/engine/sampling/sampler.py` by extracting the vectorized sampling body from `SamplingDriver.sample_batch_tokens` lines 190-269. The class should have a single public method:

```python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch

from vllm.engine.request_state import RequestState


class Sampler:
    """Vectorized temperature / top-k / top-p / multinomial sampling."""

    def sample(
        self,
        logits: torch.Tensor,
        requests: list[RequestState],
    ) -> list[int]:
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        device = logits.device
        logits = logits.float()
        batch_size, vocab_size = logits.shape

        greedy_flags = [
            float(getattr(req.sampling_params, "temperature", 0.0) or 0.0) <= 1e-6
            for req in requests
        ]
        if all(greedy_flags):
            return [int(torch.argmax(logits[i]).item()) for i in range(batch_size)]

        temps = torch.tensor(
            [float(getattr(req.sampling_params, "temperature", 0.0) or 0.0) for req in requests],
            dtype=torch.float,
            device=device,
        ).unsqueeze(1)
        top_ks = torch.tensor(
            [int(getattr(req.sampling_params, "top_k", -1) or -1) for req in requests],
            dtype=torch.long,
            device=device,
        ).unsqueeze(1)
        top_ps = torch.tensor(
            [float(getattr(req.sampling_params, "top_p", 1.0) or 1.0) for req in requests],
            dtype=torch.float,
            device=device,
        ).unsqueeze(1)

        non_greedy_mask = temps > 1e-6
        logits = torch.where(non_greedy_mask, logits / temps.clamp(min=1e-6), logits)

        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        col_indices = torch.arange(vocab_size, device=device).unsqueeze(0)
        top_k_mask = (col_indices >= top_ks) & (top_ks > 0)
        sorted_logits.masked_fill_(top_k_mask, float("-inf"))

        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(sorted_probs, dim=-1)
        sorted_remove = cumprobs > top_ps
        sorted_remove_shifted = torch.empty_like(sorted_remove)
        sorted_remove_shifted[:, 1:] = sorted_remove[:, :-1]
        sorted_remove_shifted[:, 0] = False
        sorted_logits.masked_fill_(sorted_remove_shifted, float("-inf"))

        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
        probs = torch.softmax(logits, dim=-1)

        next_tokens: list[int] = []
        for i, req in enumerate(requests):
            if greedy_flags[i]:
                next_tokens.append(int(torch.argmax(logits[i]).item()))
                continue
            row_probs = probs[i]
            psum = row_probs.sum()
            if not torch.isfinite(psum) or psum <= 0:
                next_tokens.append(int(torch.argmax(logits[i]).item()))
                continue
            generator = req.rng
            if generator is not None:
                next_tokens.append(int(torch.multinomial(row_probs, 1, generator=generator).item()))
            else:
                next_tokens.append(int(torch.multinomial(row_probs, 1).item()))
        return next_tokens
```

- [ ] **Step 4: Re-export `Sampler` from the package**

Update `vllm/engine/sampling/__init__.py`:

```python
__all__ = [
    "eos_stop_token_ids_for_sampling",
    "hf_config_eos_token_ids",
    "PenaltyEncoder",
    "Sampler",
]
```

- [ ] **Step 5: Run the tests**

```bash
uv run pytest tests/test_sampler.py -v
```

Expected: tests pass.

- [ ] **Step 6: Lint changed files**

```bash
uv run ruff check vllm/engine/sampling/ tests/test_sampler.py
```

Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add vllm/engine/sampling/ tests/test_sampler.py
git commit -m "refactor(sampling): extract vectorized Sampler"
```

---

## Task 4: Refactor `SamplingDriver` to orchestrate `PenaltyEncoder` + `Sampler`

**Files:**
- Modify: `vllm/engine/sampling_driver.py`

**Interfaces:**
- Consumes: `PenaltyEncoder`, `Sampler` from Task 2 and Task 3.
- Produces: unchanged public API on `SamplingDriver`.

- [ ] **Step 1: Write the failing integration test**

Add to `tests/test_sampler.py` or create `tests/test_sampling_driver_integration.py`:

```python
# SPDX-License-Identifier: Apache-2.0
import torch

from vllm.engine.request_state import RequestState
from vllm.engine.sampling_driver import SamplingDriver
from vllm.policies.base import GenerationPolicies
from vllm.sampling_params import SamplingParams


class FakeTokenizer:
    eos_token_id = 2


class NoOpOutputProcessor:
    def apply_context_bias(self, logits, generated_ids, sampling_params, bias_token_ids, is_capital_question):
        return logits


def test_driver_uses_vectorized_path_by_default() -> None:
    policies = GenerationPolicies(backend=NoOpOutputProcessor())
    driver = SamplingDriver(FakeTokenizer(), None, policies)
    sp = SamplingParams(temperature=0.0, repetition_penalty=1.2)
    requests = [
        RequestState(request_id="r0", prompt="", input_ids=[1], sampling_params=sp, generated_ids=[1]),
        RequestState(request_id="r1", prompt="", input_ids=[1], sampling_params=sp, generated_ids=[2]),
    ]
    logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    tokens = driver.sample_batch_tokens(logits, requests)
    assert isinstance(tokens, list)
    assert len(tokens) == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_sampling_driver_integration.py -v
```

Expected: test fails because `SamplingDriver` does not yet construct `PenaltyEncoder`/`Sampler`.

- [ ] **Step 3: Refactor `SamplingDriver`**

In `vllm/engine/sampling_driver.py`:

1. Import the new classes:

```python
import os
from vllm.engine.sampling import PenaltyEncoder, Sampler
```

2. In `__init__`, instantiate them:

```python
self._penalty_encoder = PenaltyEncoder(tokenizer, hf_config, policies)
self._sampler = Sampler()
self._use_legacy = os.environ.get("FASTINFERENCE_USE_LEGACY_SAMPLING", "0") == "1"
```

3. Replace `sample_batch_tokens` body with:

```python
def sample_batch_tokens(self, logits_2d, requests):
    if not requests:
        return []
    if self._use_legacy:
        return self._legacy_sample_batch_tokens(logits_2d, requests)

    if logits_2d.ndim == 3:
        logits_2d = logits_2d.squeeze(1)
    elif logits_2d.ndim == 1:
        logits_2d = logits_2d.unsqueeze(0)

    encoded = self._penalty_encoder.encode(logits_2d, requests)
    return self._sampler.sample(encoded, requests)
```

4. Rename the existing implementation to `_legacy_sample_batch_tokens` and keep it unchanged for the opt-out path.

5. Update `sample_next_token` to reuse `PenaltyEncoder.encode_row` + `Sampler.sample`:

```python
def sample_next_token(self, logits_1d, request):
    encoded = self._penalty_encoder.encode_row(logits_1d, request)
    return self._sampler.sample(encoded, [request])[0]
```

- [ ] **Step 4: Run integration tests**

```bash
uv run pytest tests/test_sampling_driver_integration.py -v
```

Expected: tests pass.

- [ ] **Step 5: Run existing decode-related tests**

```bash
uv run pytest tests/test_edge_cases.py tests/test_chunked_prefill_long_context.py tests/test_step_scheduler.py -q
```

Expected: all pass.

- [ ] **Step 6: Lint changed files**

```bash
uv run ruff check vllm/engine/sampling_driver.py tests/test_sampling_driver_integration.py
```

Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add vllm/engine/sampling_driver.py tests/test_sampling_driver_integration.py
git commit -m "refactor(sampling): SamplingDriver orchestrates PenaltyEncoder and Sampler"
```

---

## Task 5: Add microbenchmark and run final regression

**Files:**
- Create: `benchmarks/sampling_driver_microbenchmark.py`

**Interfaces:**
- Consumes: `SamplingDriver`, `RequestState`, `SamplingParams`.
- Produces: printed latency numbers; no code interface.

- [ ] **Step 1: Create the benchmark script**

Create `benchmarks/sampling_driver_microbenchmark.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark for SamplingDriver.sample_batch_tokens."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from vllm.engine.request_state import RequestState
from vllm.engine.sampling_driver import SamplingDriver
from vllm.policies.base import GenerationPolicies
from vllm.sampling_params import SamplingParams


class FakeTokenizer:
    eos_token_id = 2


class NoOpOutputProcessor:
    def apply_context_bias(self, logits, generated_ids, sampling_params, bias_token_ids, is_capital_question):
        return logits


@dataclass
class Config:
    batch_size: int
    vocab_size: int
    temperature: float
    repetition_penalty: float
    frequency_penalty: float
    presence_penalty: float


def make_requests(config: Config) -> list[RequestState]:
    sp = SamplingParams(
        temperature=config.temperature,
        top_k=50,
        top_p=0.9,
        repetition_penalty=config.repetition_penalty,
        frequency_penalty=config.frequency_penalty,
        presence_penalty=config.presence_penalty,
        ignore_eos=False,
        min_tokens=5,
    )
    rng = torch.Generator(device="cuda")
    requests = []
    for i in range(config.batch_size):
        generated_ids = [(i + j) % config.vocab_size for j in range(50)]
        requests.append(
            RequestState(
                request_id=f"r{i}",
                prompt="",
                input_ids=[1, 2, 3],
                sampling_params=sp,
                generated_ids=generated_ids,
                rng=rng,
            )
        )
    return requests


def bench(label: str, config: Config, warmup: int = 10, reps: int = 50) -> None:
    policies = GenerationPolicies(backend=NoOpOutputProcessor())
    driver = SamplingDriver(FakeTokenizer(), None, policies)
    requests = make_requests(config)
    for _ in range(warmup):
        logits = torch.randn(config.batch_size, config.vocab_size, device="cuda", dtype=torch.float16)
        driver.sample_batch_tokens(logits, requests)
    times = []
    for _ in range(reps):
        logits = torch.randn(config.batch_size, config.vocab_size, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        start = time.perf_counter()
        driver.sample_batch_tokens(logits, requests)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    avg = sum(times) / len(times)
    print(f"{label:20s} bs={config.batch_size:3d} vocab={config.vocab_size:6d} avg={avg*1e3:7.3f}ms")


def main() -> None:
    for vocab_size in [32000, 151936]:
        print(f"\n--- vocab_size={vocab_size} ---")
        for bs in [1, 2, 4, 8, 16]:
            bench("penalties+sample", Config(bs, vocab_size, 0.7, 1.1, 0.1, 0.1))
        for bs in [1, 8, 16]:
            bench("greedy", Config(bs, vocab_size, 0.0, 1.0, 0.0, 0.0))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the benchmark**

```bash
uv run python benchmarks/sampling_driver_microbenchmark.py
```

Expected: penalties + sample path is significantly faster than the pre-refactor baseline captured in the design doc (e.g., per-request time should drop by at least 50% for `bs >= 8`).

- [ ] **Step 3: Run the full regression suite**

```bash
bash tests/run_regression_suite.sh
```

Expected: `134 passed, 2 skipped`.

- [ ] **Step 4: Run inference correctness regression (B-tier)**

```bash
SKIP_A_TIER=1 bash tests/run_inference_correctness_regression.sh
```

Expected: exit 0, all model spotchecks PASS.

- [ ] **Step 5: Run e2e benchmark**

```bash
uv run python tests/e2e_full_benchmark.py --json-out /tmp/sampling_e2e.json
```

Expected: exit 0, no performance regression vs baseline.

- [ ] **Step 6: Format and lint**

```bash
uv run ruff format vllm/engine/sampling/ vllm/engine/sampling_driver.py tests/test_penalty_encoder.py tests/test_sampler.py tests/test_sampling_driver_integration.py benchmarks/sampling_driver_microbenchmark.py
uv run ruff check vllm/engine/sampling/ vllm/engine/sampling_driver.py tests/test_penalty_encoder.py tests/test_sampler.py tests/test_sampling_driver_integration.py tests/test_sampling_utils.py benchmarks/sampling_driver_microbenchmark.py
```

Expected: `All checks passed!`

- [ ] **Step 7: Commit the benchmark**

```bash
git add benchmarks/sampling_driver_microbenchmark.py
git commit -m "bench(sampling): add SamplingDriver microbenchmark"
```

---

## Self-Review Checklist

- [ ] Spec coverage: vectorized penalties, EOS mask, anti-template, fallback, Sampler extraction, legacy opt-out, tests, benchmark — all map to tasks above.
- [ ] No placeholders: every step has exact file paths, code snippets, commands, and expected outputs.
- [ ] Type consistency: `PenaltyEncoder.encode` returns `torch.Tensor`; `Sampler.sample` returns `list[int]`; `SamplingDriver` public API unchanged.
- [ ] DRY: EOS helpers live in one place; fallback reuses `encode_row`.
- [ ] YAGNI: no new Triton kernel, no output-pipeline changes, no scheduler changes.
