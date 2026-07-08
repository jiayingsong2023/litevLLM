# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from tests.tools.gemma4_26b_profile import run_e2e_baseline


def test_harness_imports() -> None:
    # Guard only: the harness must be importable and its helper signatures
    # must match what the tool expects. Heavy e2e execution is opt-in only.
    assert callable(run_e2e_baseline)
