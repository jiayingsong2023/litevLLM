# SPDX-License-Identifier: Apache-2.0
from vllm.engine.planners.types import DecodePlanResult, PrefillPlanResult


def test_result_classes_exist() -> None:
    assert PrefillPlanResult is not None
    assert DecodePlanResult is not None
