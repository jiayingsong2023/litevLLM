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
