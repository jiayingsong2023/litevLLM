# SPDX-License-Identifier: Apache-2.0
"""Verify that Gemma4 tuning/profile flags are instance-isolated.

After PR2, ``set_gemma4_tuning_config`` returns a ``Gemma4LayerConfig``
instance with no module-level side effects.  These tests verify that
each returned config is independent.
"""

import pytest

from vllm.model_executor.models.gemma4 import set_gemma4_tuning_config


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
