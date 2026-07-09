# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch


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
