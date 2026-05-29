# SPDX-License-Identifier: Apache-2.0
"""Verify that Gemma4 tuning/profile flags are instance-isolated.

Before global state remediation (PR2/PR3), these tests are expected to
fail (xfail) -- they demonstrate that module-level ``_GEMMA4_TUNING``,
``_GEMMA4_PROFILE_ENABLED``, and ``_GEMMA4_ROCTX_PROFILE_ENABLED``
are currently shared across instances.
"""

import pytest

from vllm.model_executor.models.gemma4 import (
    set_gemma4_tuning_config,
    _GEMMA4_PROFILE_ENABLED,
    _GEMMA4_ROCTX_PROFILE_ENABLED,
    _GEMMA4_TUNING,
)


@pytest.fixture(autouse=True)
def _reset_gemma4_global_state():
    """Reset module-level globals between tests so failures don't cascade."""
    # Reset to defaults
    set_gemma4_tuning_config(None, locked=False)
    yield
    set_gemma4_tuning_config(None, locked=False)


@pytest.mark.xfail(
    reason="PR2 not yet applied: _GEMMA4_TUNING is module-level mutable state"
)
def test_tuning_config_isolation_across_instances():
    """Two calls to set_gemma4_tuning_config with different profiles should
    be stored independently, not overwrite each other at module level."""
    set_gemma4_tuning_config(
        {"FASTINFERENCE_GEMMA4_LAYER_PROFILE": "1"}, locked=False
    )
    tuning_a = dict(_GEMMA4_TUNING)
    assert tuning_a.get("FASTINFERENCE_GEMMA4_LAYER_PROFILE") == "1"

    set_gemma4_tuning_config(
        {"FASTINFERENCE_GEMMA4_LAYER_PROFILE": "0"}, locked=False
    )
    tuning_b = dict(_GEMMA4_TUNING)

    # With proper instance isolation, tuning_a should still have "1",
    # tuning_b should have "0". Currently they share module-level dict.
    assert tuning_a.get("FASTINFERENCE_GEMMA4_LAYER_PROFILE") == "1", (
        "Expected tuning_a to retain '1' after second set_gemma4_tuning_config "
        "-- but module-level _GEMMA4_TUNING was overwritten to "
        f"'{tuning_a.get('FASTINFERENCE_GEMMA4_LAYER_PROFILE')}'"
    )
    assert tuning_b.get("FASTINFERENCE_GEMMA4_LAYER_PROFILE") == "0"


@pytest.mark.xfail(
    reason="PR2 not yet applied: _GEMMA4_PROFILE_ENABLED is module-level mutable state"
)
def test_profile_flag_isolation():
    """Two different profile configurations should have independent
    _GEMMA4_PROFILE_ENABLED flags."""
    set_gemma4_tuning_config(
        {"FASTINFERENCE_GEMMA4_LAYER_PROFILE": "1"}, locked=False
    )
    profile_a = _GEMMA4_PROFILE_ENABLED

    set_gemma4_tuning_config(
        {"FASTINFERENCE_GEMMA4_LAYER_PROFILE": "0"}, locked=False
    )
    profile_b = _GEMMA4_PROFILE_ENABLED

    assert profile_a is True
    assert profile_b is False
