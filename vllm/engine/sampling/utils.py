# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from vllm.sampling_params import SamplingParams


def hf_config_eos_token_ids(hf_config: Any | None) -> list[int]:
    """Return EOS token IDs declared by a Hugging Face model config.

    Args:
        hf_config: A Hugging Face ``PretrainedConfig`` object, or ``None``.

    Returns:
        A list of integer EOS token IDs extracted from ``eos_token_id``.
        Returns an empty list when ``hf_config`` is ``None`` or the config
        does not define an EOS token.
    """
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
    """Build the ordered, deduplicated set of EOS/stop token IDs for sampling.

    Args:
        tokenizer: A tokenizer exposing ``eos_token_id`` (int, list, tuple,
            or ``None``).
        sp: The sampling parameters for the request, which may provide
            ``stop_token_ids``.
        hf_config: Optional Hugging Face config whose EOS token IDs are also
            included.

    Returns:
        A list of integer token IDs combining the tokenizer EOS, HF config EOS,
        and ``sp.stop_token_ids``, preserving first-seen order and removing
        duplicates.
    """
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
