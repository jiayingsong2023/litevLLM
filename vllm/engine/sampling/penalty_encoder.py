# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import Counter
from typing import Any

import torch

from vllm.engine.request_state import RequestState
from vllm.engine.sampling.utils import eos_stop_token_ids_for_sampling


class PenaltyEncoder:
    """Apply sampling penalties, biases and masks to a batch of logits."""

    def __init__(self, tokenizer: Any, hf_config: Any | None, policies: Any) -> None:
        """Initialize the penalty encoder.

        Args:
            tokenizer: The tokenizer used to resolve EOS/stop token IDs.
            hf_config: The model's Hugging Face config, if available.
            policies: The generation policies backend that applies context bias
                and structured-output constraints.
        """
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
        adjusted = logits.float().clone()

        fallback_rows = self._find_fallback_rows(requests)
        vectorized_rows = [i for i in range(batch_size) if i not in fallback_rows]

        if vectorized_rows:
            self._apply_vectorized_penalties(
                adjusted, requests, vectorized_rows, vocab_size
            )

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
                for tid in eos_stop_token_ids_for_sampling(
                    self.tokenizer, sp, self.hf_config
                ):
                    if 0 <= tid < vocab_size:
                        row[tid] = float("-inf")
        else:
            for tid in eos_stop_token_ids_for_sampling(
                self.tokenizer, sp, self.hf_config
            ):
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

    def _valid_generated_ids(
        self, generated_ids: list[Any], vocab_size: int
    ) -> list[int]:
        """Return generated token IDs clamped to the valid vocabulary range."""
        return [int(t) for t in generated_ids if 0 <= int(t) < vocab_size]

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
        unique_lists = [
            list(set(self._valid_generated_ids(requests[i].generated_ids, vocab_size)))
            for i in rows
        ]
        max_len = max((len(u) for u in unique_lists), default=0)
        if max_len:
            dummy = vocab_size
            idx = torch.full(
                (len(rows), max_len), dummy, dtype=torch.long, device=device
            )
            mask = torch.zeros((len(rows), max_len), dtype=torch.bool, device=device)
            for r, uids in enumerate(unique_lists):
                if uids:
                    idx[r, : len(uids)] = torch.tensor(
                        uids, dtype=torch.long, device=device
                    )
                    mask[r, : len(uids)] = True

            padded = torch.cat(
                [row_logits, torch.zeros(len(rows), 1, device=device)], dim=1
            )
            selected = torch.gather(padded, 1, idx)
            rps = torch.tensor(
                [
                    float(
                        getattr(requests[i].sampling_params, "repetition_penalty", 1.0)
                        or 1.0
                    )
                    for i in rows
                ],
                dtype=torch.float,
                device=device,
            ).unsqueeze(1)
            factors = torch.where(selected > 0, 1.0 / rps.clamp(min=1e-6), rps)
            factors = torch.where(mask, factors, torch.ones_like(factors))

            multipliers = torch.ones(
                len(rows), vocab_size + 1, dtype=torch.float, device=device
            )
            multipliers.scatter_(1, idx, factors)
            row_logits.mul_(multipliers[:, :vocab_size])

        # --- frequency & presence penalties (additive) ---
        # Frequency
        freq_lists = []
        freq_counts = []
        max_freq = 0
        for i in rows:
            counts = Counter(
                self._valid_generated_ids(requests[i].generated_ids, vocab_size)
            )
            tids = list(counts.keys())
            cnts = [counts[t] for t in tids]
            freq_lists.append(tids)
            freq_counts.append(cnts)
            max_freq = max(max_freq, len(tids))

        if max_freq:
            dummy = vocab_size
            idx = torch.full(
                (len(rows), max_freq), dummy, dtype=torch.long, device=device
            )
            vals = torch.zeros((len(rows), max_freq), dtype=torch.float, device=device)
            fps = torch.tensor(
                [
                    float(
                        getattr(requests[i].sampling_params, "frequency_penalty", 0.0)
                        or 0.0
                    )
                    for i in rows
                ],
                dtype=torch.float,
                device=device,
            ).unsqueeze(1)
            for r, (tids, cnts) in enumerate(zip(freq_lists, freq_counts)):
                if tids:
                    idx[r, : len(tids)] = torch.tensor(
                        tids, dtype=torch.long, device=device
                    )
                    vals[r, : len(tids)] = -fps[r].item() * torch.tensor(
                        cnts, dtype=torch.float, device=device
                    )
            additive = torch.zeros(
                len(rows), vocab_size + 1, dtype=torch.float, device=device
            )
            additive.scatter_add_(1, idx, vals)
            row_logits.add_(additive[:, :vocab_size])

        # Presence
        pres_lists = [
            list(set(self._valid_generated_ids(requests[i].generated_ids, vocab_size)))
            for i in rows
        ]
        max_pres = max((len(u) for u in pres_lists), default=0)
        if max_pres:
            dummy = vocab_size
            idx = torch.full(
                (len(rows), max_pres), dummy, dtype=torch.long, device=device
            )
            pps = torch.tensor(
                [
                    float(
                        getattr(requests[i].sampling_params, "presence_penalty", 0.0)
                        or 0.0
                    )
                    for i in rows
                ],
                dtype=torch.float,
                device=device,
            ).unsqueeze(1)
            vals = -pps.expand(-1, max_pres)
            mask = torch.zeros((len(rows), max_pres), dtype=torch.bool, device=device)
            for r, uids in enumerate(pres_lists):
                if uids:
                    idx[r, : len(uids)] = torch.tensor(
                        uids, dtype=torch.long, device=device
                    )
                    mask[r, : len(uids)] = True
            vals = torch.where(mask, vals, torch.zeros_like(vals))
            additive = torch.zeros(
                len(rows), vocab_size + 1, dtype=torch.float, device=device
            )
            additive.scatter_add_(1, idx, vals)
            row_logits.add_(additive[:, :vocab_size])

        # --- EOS mask ---
        eos_lists = [
            eos_stop_token_ids_for_sampling(
                self.tokenizer, requests[i].sampling_params, self.hf_config
            )
            for i in rows
        ]
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
