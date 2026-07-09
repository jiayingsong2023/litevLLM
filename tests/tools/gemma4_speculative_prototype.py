# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch

from vllm import LLM
from vllm.sampling_params import SamplingParams


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


def baseline_greedy(
    llm: LLM, prompt_text: str, max_new_tokens: int
) -> tuple[list[int], list[int]]:
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

    ``baseline_token_ids`` and ``bit_exact`` are caller-filled placeholders;
    the offline CLI populates them after the baseline run.
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
        accepted_total += sum(
            1 for tok, draft in zip(accepted[: len(proposed)], proposed) if tok == draft
        )
        proposed_total += len(proposed)

        if len(generated) >= max_new_tokens:
            generated = generated[:max_new_tokens]
            break
    elapsed = time.perf_counter() - start_time

    acceptance_rate = accepted_total / proposed_total if proposed_total > 0 else 0.0
    speculative_tps = len(generated) / elapsed if elapsed > 0 else 0.0

    return {
        "token_ids": generated,
        "baseline_token_ids": [],  # caller-filled placeholder
        "bit_exact": False,  # caller-filled placeholder
        "accepted_total": accepted_total,
        "proposed_total": proposed_total,
        "acceptance_rate": acceptance_rate,
        "target_forwards": target_forwards,
        "baseline_tps": 0.0,  # filled by CLI caller
        "speculative_tps": speculative_tps,
        "projected_tps": 0.0,  # filled by CLI caller
    }
