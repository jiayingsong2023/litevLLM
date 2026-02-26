# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from collections.abc import Iterable, Iterator, MutableSequence
from dataclasses import dataclass, field
from typing import overload

# We use dataclass for now because it is used for
# openai server output, and msgspec is not serializable.
# TODO(sang): Fix it.
@dataclass
class Logprob:

    logprob: float
    rank: int | None = None
    decoded_token: str | None = None

LogprobsOnePosition = dict[int, Logprob]

@dataclass
class FlatLogprobs(MutableSequence[LogprobsOnePosition | None]):

    # Start / end indices to indicate the range of logprobs for each position.
    start_indices: list[int] = field(default_factory=list)
    end_indices: list[int] = field(default_factory=list)

    # Flatten Logprob information for (each position, rank).
    # For position <i>, the logprobs are ranged
    # from self.start_indices[i] to self.end_indices[i] (exclusive).
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    ranks: list[int | None] = field(default_factory=list)
    decoded_tokens: list[str | None] = field(default_factory=list)

    def append(self, logprobs_one_position: LogprobsOnePosition | None) -> None:
        Appends logprobs for the next position without creating
        the intermediate logprob dictionary.
        for logprobs_one_position in logprobs_multi_positions:
            self.append(logprobs_one_position)

    def __len__(self) -> int:
        if isinstance(index, int):
            return {
                self.token_ids[i]: Logprob(
                    logprob=self.logprobs[i],
                    rank=self.ranks[i],
                    decoded_token=self.decoded_tokens[i],
                )
                for i in range(self.start_indices[index], self.end_indices[index])
            }
        elif isinstance(index, slice):
            min_index = self.start_indices[index][0]
            max_index = self.end_indices[index][-1]
            return FlatLogprobs(
                # Shift updated start_indices and end_indices to
                # be 0-indexed
                start_indices=[i - min_index for i in self.start_indices[index]],
                end_indices=[i - min_index for i in self.end_indices[index]],
                token_ids=self.token_ids[min_index:max_index],
                logprobs=self.logprobs[min_index:max_index],
                ranks=self.ranks[min_index:max_index],
                decoded_tokens=self.decoded_tokens[min_index:max_index],
            )
        else:
            raise TypeError(f"Invalid index type: {type(index)}")

    def __setitem__(self, item, value) -> None:
        raise TypeError("Cannot set logprobs in FlatLogprobs")

    def __delitem__(self, item) -> None:
        raise TypeError("Cannot delete logprobs from FlatLogprobs")

    def insert(self, index: int, value: dict[int, Logprob] | None) -> None:
        raise TypeError("Cannot insert logprobs to FlatLogprobs")

    def __iter__(self) -> Iterator[LogprobsOnePosition]:
        for i in range(0, len(self.start_indices)):
            yield self.__getitem__(i)

# {token_id -> logprob} per each sequence group. None if the corresponding
# sequence group doesn't require prompt logprob.
PromptLogprobs = FlatLogprobs | list[LogprobsOnePosition | None]
# {token_id -> logprob} for each sequence group.
SampleLogprobs = FlatLogprobs | list[LogprobsOnePosition]

def create_prompt_logprobs(flat_logprobs: bool) -> PromptLogprobs:
    return FlatLogprobs() if flat_logprobs else []

def append_logprobs_for_next_position(
    request_logprobs: PromptLogprobs | SampleLogprobs,
    token_ids: list[int],
    logprobs: list[float],
    decoded_tokens: Iterable[str | None],
    rank: int,
    num_logprobs: int,
) -> None:
