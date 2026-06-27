# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from typing import Any


class CompletionOutput:
    """The output data of one completion output of a request.

    ``text`` is decoded lazily when a tokenizer is supplied and ``text`` was
    not provided at construction time. This avoids paying the ``tokenizer.decode``
    cost for intermediate outputs that are never consumed by the caller
    (e.g. non-streaming requests that only use the final output).
    """

    def __init__(
        self,
        index: int,
        text: str | None,
        token_ids: list[int],
        cumulative_logprob: float,
        tokenizer: Any = None,
        sampling_params: Any = None,
        text_processor: Callable[[str], str] | None = None,
        finished: bool = False,
        **kwargs,
    ):
        self.index = index
        self._text = text
        self._tokenizer = tokenizer
        self._sampling_params = sampling_params
        self._text_processor = text_processor
        self._finished = finished
        self.token_ids = token_ids
        self.cumulative_logprob = cumulative_logprob

    @property
    def text(self) -> str:
        if self._text is not None:
            return self._text
        if self._tokenizer is None:
            return ""
        decoded = self._tokenizer.decode(
            self.token_ids,
            skip_special_tokens=getattr(
                self._sampling_params, "skip_special_tokens", True
            ),
            spaces_between_special_tokens=getattr(
                self._sampling_params, "spaces_between_special_tokens", False
            ),
            clean_up_tokenization_spaces=True,
        )
        if self._text_processor is not None:
            decoded = self._text_processor(decoded)
        if self._finished:
            self._text = decoded
        return decoded

    @text.setter
    def text(self, value: str | None) -> None:
        self._text = value


class RequestOutput:
    """The output data of a request to the LLM."""

    def __init__(
        self,
        request_id: str,
        prompt: str,
        prompt_token_ids: list[int],
        outputs: list[CompletionOutput],
        finished: bool,
        **kwargs,
    ):
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
        prompt_token_ids: list[int] | None = None,
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
    def __init__(self, probs: list[float]):
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
            score = float(data[0]) if isinstance(data, list) and data else 0.0
        return cls(ScoringOutput(float(score)))
