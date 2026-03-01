# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import List, Optional, Union

@dataclass
class StructuredOutputsParams:
    json: Optional[str] = None
    regex: Optional[str] = None
    grammar: Optional[str] = None
    structural_tag: Optional[str] = None
    json_object: bool = False
    choice: Optional[List[str]] = None

class RequestOutputKind:
    DELTA = "delta"
    FINAL_ONLY = "final_only"

@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    max_tokens: Optional[int] = 16
    min_tokens: int = 0
    stop: List[str] = field(default_factory=list)
    stop_token_ids: List[int] = field(default_factory=list)
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    ignore_eos: bool = False
    n: int = 1
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    seed: Optional[int] = None
    include_stop_str_in_output: bool = False
    logit_bias: Optional[dict] = None
    truncate_prompt_tokens: Optional[int] = None
    structured_outputs: Optional[StructuredOutputsParams] = None
    detokenize: bool = True
    output_kind: str = RequestOutputKind.FINAL_ONLY
