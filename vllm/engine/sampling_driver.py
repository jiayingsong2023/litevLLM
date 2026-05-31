# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import Counter
from typing import Any

import torch

from vllm.sampling_params import SamplingParams


def hf_config_eos_token_ids(hf_config: Any | None) -> list[int]:
    if hf_config is None:
        return []
    eos = getattr(hf_config, "eos_token_id", None)
    if eos is None:
        return []
    if isinstance(eos, (list, tuple)):
        return [int(x) for x in eos]
    return [int(eos)]


def eos_stop_token_ids_for_sampling(
    tokenizer: Any,
    sp: SamplingParams,
    hf_config: Any | None = None,
) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()

    def _add(tid: int) -> None:
        if tid not in seen:
            seen.add(tid)
            out.append(tid)

    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        if isinstance(eos, (list, tuple)):
            for x in eos:
                _add(int(x))
        else:
            _add(int(eos))
    for tid in hf_config_eos_token_ids(hf_config):
        _add(tid)
    for tid in getattr(sp, "stop_token_ids", None) or ():
        _add(int(tid))
    return out


class SamplingDriver:
    def __init__(self, tokenizer: Any, hf_config: Any | None, policies: Any):
        self.tokenizer = tokenizer
        self.hf_config = hf_config
        self.policies = policies

    def sample_next_token(
        self,
        logits_1d: torch.Tensor,
        request: dict[str, Any],
    ) -> int:
        token_logits = self.policies.apply_context_bias(
            logits_1d,
            request["generated_ids"],
            request["sampling_params"],
            request.get("capital_question_bias_token_ids"),
            request.get("is_chinese_capital_question"),
        )
        eos_mask = eos_stop_token_ids_for_sampling(
            self.tokenizer, request["sampling_params"], self.hf_config
        )
        structured_output_constraint = request.get("structured_output_constraint")
        if structured_output_constraint is not None:
            token_logits = structured_output_constraint.apply(token_logits, request)
        return self._sample_token_from_logits(
            token_logits,
            request["sampling_params"],
            request["generated_ids"],
            request.get("rng"),
            eos_token_ids_to_mask=eos_mask,
            anti_template_token_ids=request.get("anti_template_token_ids"),
        )

    def completion_eos_ids(self, request: dict[str, Any]) -> list[int]:
        return eos_stop_token_ids_for_sampling(
            self.tokenizer, request["sampling_params"], self.hf_config
        )

    def sample_batch_tokens(
        self,
        logits_2d: torch.Tensor,
        requests: list[dict[str, Any]],
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
            sp = req["sampling_params"]

            # Apply context bias
            row_logits = self.policies.apply_context_bias(
                logits[i],
                req["generated_ids"],
                sp,
                req.get("capital_question_bias_token_ids"),
                req.get("is_chinese_capital_question"),
            )

            # Apply structured output constraint
            structured_output_constraint = req.get("structured_output_constraint")
            if structured_output_constraint is not None:
                row_logits = structured_output_constraint.apply(row_logits, req)

            # Apply repetition penalty
            rp = float(getattr(sp, "repetition_penalty", 1.0) or 1.0)
            generated_ids = req["generated_ids"]
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
            anti_template = req.get("anti_template_token_ids")
            if anti_template and len(generated_ids) < 12:
                for tid in anti_template:
                    if 0 <= tid < VocabSize:
                        row_logits[tid] -= 60.0

            logits[i] = row_logits

        greedy_flags = [
            float(getattr(req["sampling_params"], "temperature", 0.0) or 0.0) <= 1e-6
            for req in requests
        ]
        if all(greedy_flags):
            return [int(torch.argmax(logits[i]).item()) for i in range(B)]

        # Extract parameters into tensors
        temps = torch.tensor(
            [
                float(getattr(req["sampling_params"], "temperature", 0.0) or 0.0)
                for req in requests
            ],
            dtype=torch.float,
            device=device,
        ).unsqueeze(1)
        top_ks = torch.tensor(
            [
                int(getattr(req["sampling_params"], "top_k", -1) or -1)
                for req in requests
            ],
            dtype=torch.long,
            device=device,
        ).unsqueeze(1)
        top_ps = torch.tensor(
            [
                float(getattr(req["sampling_params"], "top_p", 1.0) or 1.0)
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

            generator = req.get("rng")
            if generator is not None:
                token = int(torch.multinomial(row_probs, 1, generator=generator).item())
            else:
                token = int(torch.multinomial(row_probs, 1).item())
            next_tokens.append(token)

        return next_tokens

    @staticmethod
    def _sample_token_from_logits(
        logits_1d: torch.Tensor,
        sp: SamplingParams,
        generated_ids: list[int],
        generator: torch.Generator | None,
        eos_token_ids_to_mask: list[int] | None = None,
        anti_template_token_ids: list[int] | None = None,
    ) -> int:
        logits = logits_1d.float().clone()
        rp = float(getattr(sp, "repetition_penalty", 1.0) or 1.0)
        if rp > 1.0 and generated_ids:
            for tid in set(int(t) for t in generated_ids):
                if 0 <= tid < logits.numel():
                    if logits[tid] > 0:
                        logits[tid] /= rp
                    else:
                        logits[tid] *= rp

        fp = float(getattr(sp, "frequency_penalty", 0.0) or 0.0)
        if abs(fp) > 1e-12 and generated_ids:
            for tid, cnt in Counter(int(t) for t in generated_ids).items():
                if 0 <= tid < logits.numel():
                    logits[tid] -= fp * float(cnt)

        pp = float(getattr(sp, "presence_penalty", 0.0) or 0.0)
        if abs(pp) > 1e-12 and generated_ids:
            for tid in set(int(t) for t in generated_ids):
                if 0 <= tid < logits.numel():
                    logits[tid] -= pp

        if not getattr(sp, "ignore_eos", False) and eos_token_ids_to_mask:
            mt = int(getattr(sp, "min_tokens", 0) or 0)
            if len(generated_ids) < mt:
                for tid in eos_token_ids_to_mask:
                    if 0 <= tid < logits.numel():
                        logits[tid] = float("-inf")
        elif getattr(sp, "ignore_eos", False) and eos_token_ids_to_mask:
            for tid in eos_token_ids_to_mask:
                if 0 <= tid < logits.numel():
                    logits[tid] = float("-inf")

        if anti_template_token_ids and len(generated_ids) < 12:
            for tid in anti_template_token_ids:
                if 0 <= tid < logits.numel():
                    logits[tid] -= 60.0

        temp = float(getattr(sp, "temperature", 0.0) or 0.0)
        if temp <= 1e-6:
            return int(torch.argmax(logits).item())

        logits = logits / max(temp, 1e-6)

        top_k = int(getattr(sp, "top_k", -1) or -1)
        if 0 < top_k < logits.numel():
            k = min(top_k, logits.numel())
            _, top_idx = torch.topk(logits, k)
            keep = torch.zeros_like(logits, dtype=torch.bool)
            keep[top_idx] = True
            logits = logits.masked_fill(~keep, float("-inf"))

        top_p = float(getattr(sp, "top_p", 1.0) or 1.0)
        if top_p < 1.0 - 1e-6:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(sorted_probs, dim=-1)
            sorted_remove = cumprobs > top_p
            sorted_remove[1:] = sorted_remove[:-1].clone()
            sorted_remove[0] = False
            if sorted_remove.all():
                sorted_remove[:] = True
                sorted_remove[0] = False
            removed_idx = sorted_indices[sorted_remove]
            logits = logits.clone()
            logits[removed_idx] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        psum = probs.sum()
        if not torch.isfinite(psum) or psum <= 0:
            return int(torch.argmax(logits_1d.float()).item())
        if generator is not None:
            return int(torch.multinomial(probs, 1, generator=generator).item())
        return int(torch.multinomial(probs, 1).item())
