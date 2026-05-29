# SPDX-License-Identifier: Apache-2.0
"""Verify that Gemma4 tuning/profile flags are instance-isolated.

After PR2, ``set_gemma4_tuning_config`` returns a ``Gemma4LayerConfig``
instance with no module-level side effects.  These tests verify that
each returned config is independent.
"""

import pytest
import torch

from vllm.model_executor.models.gemma4 import (
    Gemma4LayerConfig,
    set_gemma4_tuning_config,
)


@pytest.fixture(autouse=True)
def _reset_gemma4_global_state():
    """Reset module-level globals between tests so failures don't cascade."""
    # Reset to defaults (triggers _apply_global_tuning_config internally)
    set_gemma4_tuning_config(None, locked=False)
    yield
    set_gemma4_tuning_config(None, locked=False)


def test_tuning_config_isolation_across_instances():
    """Two calls to set_gemma4_tuning_config with different profiles should
    return independent Gemma4LayerConfig instances."""
    config_a = set_gemma4_tuning_config(
        {"FASTINFERENCE_GEMMA4_LAYER_PROFILE": "1"}, locked=False
    )
    assert config_a.profile_enabled is True

    config_b = set_gemma4_tuning_config(
        {"FASTINFERENCE_GEMMA4_LAYER_PROFILE": "0"}, locked=False
    )
    assert config_b.profile_enabled is False

    # Instance isolation: config_a unchanged
    assert config_a.profile_enabled is True
    assert config_b.profile_enabled is False


def test_profile_flag_isolation():
    """Two different profile configurations should have independent
    profile_enabled flags."""
    config_a = set_gemma4_tuning_config(
        {"FASTINFERENCE_GEMMA4_LAYER_PROFILE": "1"}, locked=False
    )
    config_b = set_gemma4_tuning_config(
        {"FASTINFERENCE_GEMMA4_LAYER_PROFILE": "0"}, locked=False
    )
    assert config_a.profile_enabled is True
    assert config_b.profile_enabled is False


def test_rope_cache_isolation_across_configs():
    """Two Gemma4LayerConfig instances should have independent rope caches."""
    config_a = Gemma4LayerConfig()
    config_b = Gemma4LayerConfig()

    # Simulate a cache entry with a unique key
    key_a = (0, 128, 1.0, "default", 10000.0, "linear", 4096, "llama3")
    config_a.rope_cache_pool[key_a] = (
        torch.randn(1, 128), torch.randn(1, 128)
    )

    assert key_a in config_a.rope_cache_pool
    assert key_a not in config_b.rope_cache_pool, (
        "config_b's rope cache should be independent of config_a's"
    )
