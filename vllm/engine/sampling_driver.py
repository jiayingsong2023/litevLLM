# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import Counter
from typing import Any, List, Optional

import torch

from vllm.sampling_params import SamplingParams


def hf_config_eos_token_ids(hf_config: Optional[Any]) -> List[int]:
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
    hf_config: Optional[Any] = None,
) -> List[int]:
    out: List[int] = []
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
    def __init__(self, tokenizer: Any, hf_config: Optional[Any], policies: Any):
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
        return self._sample_token_from_logits(
            token_logits,
            request["sampling_params"],
            request["generated_ids"],
            request.get("rng"),
            eos_token_ids_to_mask=eos_mask,
            anti_template_token_ids=request.get("anti_template_token_ids"),
        )

    def completion_eos_ids(self, request: dict[str, Any]) -> List[int]:
        return eos_stop_token_ids_for_sampling(
            self.tokenizer, request["sampling_params"], self.hf_config
        )

    @staticmethod
    def _sample_token_from_logits(
        logits_1d: torch.Tensor,
        sp: SamplingParams,
        generated_ids: List[int],
        generator: Optional[torch.Generator],
        eos_token_ids_to_mask: Optional[List[int]] = None,
        anti_template_token_ids: Optional[List[int]] = None,
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
