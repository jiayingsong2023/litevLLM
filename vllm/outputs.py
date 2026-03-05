# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional, Any

class CompletionOutput:
    """The output data of one completion output of a request."""
    def __init__(self, index: int, text: str, token_ids: List[int], cumulative_logprob: float, **kwargs):
        self.index = index
        self.text = text
        self.token_ids = token_ids
        self.cumulative_logprob = cumulative_logprob

class RequestOutput:
    """The output data of a request to the LLM."""
    def __init__(self, request_id: str, prompt: str, prompt_token_ids: List[int], outputs: List[CompletionOutput], finished: bool, **kwargs):
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.outputs = outputs
        self.finished = finished


class PoolingOutput:
    def __init__(self, data: Any = None):
        self.data = data


class PoolingRequestOutput:
    def __init__(
        self,
        request_id: str = "",
        prompt_token_ids: Optional[List[int]] = None,
        outputs: Any = None,
        num_cached_tokens: int = 0,
        finished: bool = True,
    ):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids or []
        self.outputs = outputs if outputs is not None else PoolingOutput([])
        self.num_cached_tokens = num_cached_tokens
        self.finished = finished


class ClassificationOutput:
    def __init__(self, probs: List[float]):
        self.probs = probs

    @classmethod
    def from_base(cls, outputs: Any) -> "ClassificationOutput":
        if hasattr(outputs, "probs"):
            return cls(list(outputs.probs))
        data = getattr(outputs, "data", None)
        if isinstance(data, list) and data:
            return cls([float(x) for x in data])
        return cls([1.0])


class ScoringOutput:
    def __init__(self, score: float):
        self.score = score


class ScoringRequestOutput:
    def __init__(self, outputs: ScoringOutput):
        self.outputs = outputs

    @classmethod
    def from_base(cls, base: PoolingRequestOutput) -> "ScoringRequestOutput":
        score = getattr(getattr(base, "outputs", None), "score", None)
        if score is None:
            data = getattr(getattr(base, "outputs", None), "data", None)
            if isinstance(data, list) and data:
                score = float(data[0])
            else:
                score = 0.0
        return cls(ScoringOutput(float(score)))
