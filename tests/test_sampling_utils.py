# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock

from vllm.engine.sampling.utils import (
    eos_stop_token_ids_for_sampling,
    hf_config_eos_token_ids,
)
from vllm.sampling_params import SamplingParams


def test_hf_config_eos_token_ids_empty_when_none() -> None:
    assert hf_config_eos_token_ids(None) == []


def test_hf_config_eos_token_ids_scalar() -> None:
    config = Mock(eos_token_id=42)
    assert hf_config_eos_token_ids(config) == [42]


def test_hf_config_eos_token_ids_list() -> None:
    config = Mock(eos_token_id=[1, 2])
    assert hf_config_eos_token_ids(config) == [1, 2]


def test_hf_config_eos_token_ids_missing() -> None:
    config = Mock(spec=[])
    assert hf_config_eos_token_ids(config) == []


def test_eos_stop_token_ids_scalar_tokenizer_eos() -> None:
    tokenizer = Mock()
    tokenizer.eos_token_id = 99
    sp = SamplingParams()
    assert eos_stop_token_ids_for_sampling(tokenizer, sp, None) == [99]


def test_eos_stop_token_ids_deduplicates_sources() -> None:
    tokenizer = Mock()
    tokenizer.eos_token_id = [1, 2]
    sp = SamplingParams(stop_token_ids=[2, 3])
    assert eos_stop_token_ids_for_sampling(tokenizer, sp, None) == [1, 2, 3]


def test_eos_stop_token_ids_includes_hf_config_eos() -> None:
    tokenizer = Mock()
    tokenizer.eos_token_id = [1]
    sp = SamplingParams(stop_token_ids=[3])
    hf_config = Mock(eos_token_id=[2, 1])
    assert eos_stop_token_ids_for_sampling(tokenizer, sp, hf_config) == [1, 2, 3]


def test_eos_stop_token_ids_all_empty() -> None:
    tokenizer = Mock()
    tokenizer.eos_token_id = None
    sp = SamplingParams()
    assert eos_stop_token_ids_for_sampling(tokenizer, sp, None) == []
