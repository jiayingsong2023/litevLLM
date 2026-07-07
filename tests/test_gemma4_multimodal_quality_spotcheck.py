# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from tests.tools.gemma4_multimodal_quality_spotcheck import (
    DEFAULT_CASES,
    create_fixture_images,
    score_output,
)


def test_score_output_matches_keywords_case_insensitively() -> None:
    assert score_output("A RED square.", ("red",))
    assert not score_output("A blue circle.", ("red",))


def test_create_fixture_images_writes_all_default_images(tmp_path) -> None:
    create_fixture_images(tmp_path)

    for case in DEFAULT_CASES:
        assert (tmp_path / case.image_name).exists()
