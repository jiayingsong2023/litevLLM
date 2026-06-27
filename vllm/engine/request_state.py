# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.sampling_params import SamplingParams


@dataclass(slots=True)
class RequestState:
    """Typed request state for the lite engine control plane.

    The previous control plane passed request state around as a ``dict``.  This
    dataclass keeps the same field names and runtime semantics while making the
    shape of the state visible to type checkers.  A small dict-like shim
    (``get``/``__getitem__``/``__setitem__``/``setdefault``) is retained during
    the transition so callers that still use string-key access continue to work.
    """

    request_id: str
    prompt: str
    input_ids: list[int]
    sampling_params: SamplingParams
    device: str | None = None

    generated_ids: list[int] = field(default_factory=list)
    seq_len: int = 0
    slot_idx: int | None = None
    is_prefill: bool = True
    finished: bool = False

    queued_at: float = 0.0
    admitted_at: float | None = None
    first_token_at: float | None = None

    multi_modal_data: dict[str, Any] | None = None
    multi_modal_inputs: Any | None = None
    lora_id: str | None = None
    lora_int_id: int | None = None
    lora_path: str | None = None
    service_class: str = "latency"

    prefix_hit_len: int = 0
    _prefix_cache_entry: Any | None = None
    _prefix_cache_hit_len: int = 0
    _prefix_cache_applied: bool = False
    _prefix_cache_observed: bool = False

    low_info_hits: int = 0
    _last_token_tensor: Any | None = None
    _pending_token_tensors: list[Any] = field(default_factory=list)

    linear_attn_carry: list[Any] | None = None
    linear_conv_carry: list[Any] | None = None

    structured_output_constraint: Any | None = None
    anti_template_token_ids: list[int] | None = None
    capital_question_bias_token_ids: list[int] | None = None
    is_chinese_capital_question: bool = False

    # Fields introduced after the original typing plan but still required by the
    # current runtime (request builder, sampling, multimodal scheduling).
    guarded_prompt: str = ""
    rng: torch.Generator | None = None
    is_multimodal: bool = False
    is_multimodal_lora: bool = False

    def __post_init__(self) -> None:
        # Derive multimodal flags from the attached data when not set explicitly.
        has_image = bool((self.multi_modal_data or {}).get("image"))
        if not self.is_multimodal and has_image:
            self.is_multimodal = True
        if not self.is_multimodal_lora and self.is_multimodal and bool(self.lora_id):
            self.is_multimodal_lora = True

    # Backwards-compatible dict-like accessors --------------------------------

    def get(self, key: str, default: Any | None = None) -> Any:
        """Return the attribute named ``key`` if it exists, else ``default``."""
        try:
            return getattr(self, key)
        except AttributeError:
            return default

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def setdefault(self, key: str, default: Any) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            setattr(self, key, default)
            return default
