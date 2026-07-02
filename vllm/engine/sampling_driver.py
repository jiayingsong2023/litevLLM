# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from collections import Counter
from typing import Any

import torch

from vllm.engine.request_state import RequestState
from vllm.engine.sampling import PenaltyEncoder, Sampler
from vllm.engine.sampling.utils import eos_stop_token_ids_for_sampling


class SamplingDriver:
    """High-level sampling coordinator for the lite engine.

    Prepares logits via :class:`PenaltyEncoder` and delegates sampling to
    :class:`Sampler`. The default path vectorizes penalties and sampling across
    a batch; the legacy path (enabled via ``FASTINFERENCE_USE_LEGACY_SAMPLING=1``)
    processes rows individually and is kept for parity verification.
    """

    def __init__(self, tokenizer: Any, hf_config: Any | None, policies: Any) -> None:
        self.tokenizer = tokenizer
        self.hf_config = hf_config
        self.policies = policies
        self._penalty_encoder = PenaltyEncoder(tokenizer, hf_config, policies)
        self._sampler = Sampler()
        self._use_legacy = (
            os.environ.get("FASTINFERENCE_USE_LEGACY_SAMPLING", "0") == "1"
        )

    def sample_next_token(
        self,
        logits_1d: torch.Tensor,
        request: RequestState,
    ) -> int:
        encoded = self._penalty_encoder.encode_row(logits_1d, request)
        return self._sampler.sample(encoded, [request])[0]

    def completion_eos_ids(self, request: RequestState) -> list[int]:
        return eos_stop_token_ids_for_sampling(
            self.tokenizer, request.sampling_params, self.hf_config
        )

    def sample_batch_tokens(
        self,
        logits_2d: torch.Tensor,
        requests: list[RequestState],
    ) -> list[int]:
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

    def _legacy_sample_batch_tokens(
        self,
        logits_2d: torch.Tensor,
        requests: list[RequestState],
    ) -> list[int]:
        if not requests:
            return []

        if logits_2d.ndim == 3:
            logits_2d = logits_2d.squeeze(1)
        elif logits_2d.ndim == 1:
            logits_2d = logits_2d.unsqueeze(0)

        device = logits_2d.device
        logits = logits_2d.float().clone()
        B, VocabSize = logits.shape

        for i, req in enumerate(requests):
            sp = req.sampling_params

            # Apply context bias
            row_logits = self.policies.apply_context_bias(
                logits[i],
                req.generated_ids,
                sp,
                req.capital_question_bias_token_ids,
                req.is_chinese_capital_question,
            )

            # Apply structured output constraint
            structured_output_constraint = req.structured_output_constraint
            if structured_output_constraint is not None:
                row_logits = structured_output_constraint.apply(row_logits, req)

            # Apply repetition penalty
            rp = float(getattr(sp, "repetition_penalty", 1.0) or 1.0)
            generated_ids = req.generated_ids
            if rp > 1.0 and generated_ids:
                unique_ids = list(set(int(t) for t in generated_ids))
                valid_ids = [tid for tid in unique_ids if 0 <= tid < VocabSize]
                if valid_ids:
                    tids_tensor = torch.tensor(
                        valid_ids, dtype=torch.long, device=device
                    )
                    selected = row_logits[tids_tensor]
                    row_logits[tids_tensor] = torch.where(
                        selected > 0, selected / rp, selected * rp
                    )

            # Apply frequency penalty
            fp = float(getattr(sp, "frequency_penalty", 0.0) or 0.0)
            if abs(fp) > 1e-12 and generated_ids:
                counts = Counter(int(t) for t in generated_ids)
                tids = []
                cnts = []
                for tid, cnt in counts.items():
                    if 0 <= tid < VocabSize:
                        tids.append(tid)
                        cnts.append(cnt)
                if tids:
                    tids_tensor = torch.tensor(tids, dtype=torch.long, device=device)
                    cnts_tensor = torch.tensor(cnts, dtype=torch.float, device=device)
                    row_logits[tids_tensor] -= fp * cnts_tensor

            # Apply presence penalty
            pp = float(getattr(sp, "presence_penalty", 0.0) or 0.0)
            if abs(pp) > 1e-12 and generated_ids:
                unique_ids = list(set(int(t) for t in generated_ids))
                valid_ids = [tid for tid in unique_ids if 0 <= tid < VocabSize]
                if valid_ids:
                    tids_tensor = torch.tensor(
                        valid_ids, dtype=torch.long, device=device
                    )
                    row_logits[tids_tensor] -= pp

            # Apply EOS masking
            eos_mask = eos_stop_token_ids_for_sampling(
                self.tokenizer, sp, self.hf_config
            )
            ignore_eos = getattr(sp, "ignore_eos", False)
            if eos_mask:
                if not ignore_eos:
                    mt = int(getattr(sp, "min_tokens", 0) or 0)
                    if len(generated_ids) < mt:
                        for tid in eos_mask:
                            if 0 <= tid < VocabSize:
                                row_logits[tid] = float("-inf")
                else:
                    for tid in eos_mask:
                        if 0 <= tid < VocabSize:
                            row_logits[tid] = float("-inf")

            # Anti-template masking
            anti_template = req.anti_template_token_ids
            if anti_template and len(generated_ids) < 12:
                for tid in anti_template:
                    if 0 <= tid < VocabSize:
                        row_logits[tid] -= 60.0

            logits[i] = row_logits

        greedy_flags = [
            float(getattr(req.sampling_params, "temperature", 0.0) or 0.0) <= 1e-6
            for req in requests
        ]
        if all(greedy_flags):
            return [int(torch.argmax(logits[i]).item()) for i in range(B)]

        # Extract parameters into tensors
        temps = torch.tensor(
            [
                float(getattr(req.sampling_params, "temperature", 0.0) or 0.0)
                for req in requests
            ],
            dtype=torch.float,
            device=device,
        ).unsqueeze(1)
        top_ks = torch.tensor(
            [int(getattr(req.sampling_params, "top_k", -1) or -1) for req in requests],
            dtype=torch.long,
            device=device,
        ).unsqueeze(1)
        top_ps = torch.tensor(
            [
                float(getattr(req.sampling_params, "top_p", 1.0) or 1.0)
                for req in requests
            ],
            dtype=torch.float,
            device=device,
        ).unsqueeze(1)

        # Vectorized Temperature Scaling
        non_greedy_mask = temps > 1e-6
        logits = torch.where(non_greedy_mask, logits / temps.clamp(min=1e-6), logits)

        # Vectorized Sorting
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

        # Vectorized Top-K Filtering
        col_indices = torch.arange(VocabSize, device=device).unsqueeze(
            0
        )  # [1, VocabSize]
        top_k_mask = (col_indices >= top_ks) & (top_ks > 0)
        sorted_logits.masked_fill_(top_k_mask, float("-inf"))

        # Vectorized Top-P Filtering
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(sorted_probs, dim=-1)
        sorted_remove = cumprobs > top_ps

        sorted_remove_shifted = torch.empty_like(sorted_remove)
        sorted_remove_shifted[:, 1:] = sorted_remove[:, :-1]
        sorted_remove_shifted[:, 0] = False

        sorted_logits.masked_fill_(sorted_remove_shifted, float("-inf"))

        # Unsort/Scatter back to logits
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

        # Sample final tokens
        probs = torch.softmax(logits, dim=-1)

        next_tokens: list[int] = []
        for i, req in enumerate(requests):
            temp = temps[i].item()
            if temp <= 1e-6:
                next_tokens.append(int(torch.argmax(logits[i]).item()))
                continue

            row_probs = probs[i]
            psum = row_probs.sum()
            if not torch.isfinite(psum) or psum <= 0:
                next_tokens.append(int(torch.argmax(logits_2d[i].float()).item()))
                continue

            generator = req.rng
            if generator is not None:
                token = int(torch.multinomial(row_probs, 1, generator=generator).item())
            else:
                token = int(torch.multinomial(row_probs, 1).item())
            next_tokens.append(token)

        return next_tokens


