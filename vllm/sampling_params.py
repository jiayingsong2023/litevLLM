# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field


@dataclass
class StructuredOutputsParams:
    json: str | dict | None = None
    regex: str | None = None
    grammar: str | None = None
    structural_tag: str | None = None
    json_object: bool = False
    choice: list[str] | None = None

    def all_constraints_none(self) -> bool:
        return not any(
            (
                self.json is not None,
                self.regex is not None,
                self.grammar is not None,
                self.structural_tag is not None,
                self.json_object,
                self.choice is not None,
            )
        )


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
    max_tokens: int | None = 16
    min_tokens: int = 0
    stop: list[str] = field(default_factory=list)
    stop_token_ids: list[int] = field(default_factory=list)
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    ignore_eos: bool = False
    n: int = 1
    logprobs: int | None = None
    prompt_logprobs: int | None = None
    seed: int | None = None
    include_stop_str_in_output: bool = False
    logit_bias: dict | None = None
    truncate_prompt_tokens: int | None = None
    structured_outputs: StructuredOutputsParams | None = None
    detokenize: bool = True
    output_kind: str = RequestOutputKind.FINAL_ONLY


@dataclass
class BeamSearchParams:
    beam_width: int = 1
    max_tokens: int | None = 16
    ignore_eos: bool = False
