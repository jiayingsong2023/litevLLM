# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock

from vllm.engine.sampling.utils import eos_stop_token_ids_for_sampling
from vllm.sampling_params import SamplingParams


def test_eos_stop_token_ids_deduplicates_sources() -> None:
    tokenizer = Mock()
    tokenizer.eos_token_id = [1, 2]
    sp = SamplingParams(stop_token_ids=[2, 3])
    assert eos_stop_token_ids_for_sampling(tokenizer, sp, None) == [1, 2, 3]
