# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from pathlib import Path

from vllm import envs

class NewLineFormatter(logging.Formatter):
            Shortens a file path for logging display:
            - Removes leading 'vllm' folder if present.
            - If path starts with 'v1',
            keeps the first two and last two levels,
            collapsing the middle as '...'.
            - Otherwise, keeps the first and last two levels,
            collapsing the middle as '...'.
            - If the path is short, returns it as-is.
            - Examples:
            vllm/model_executor/layers/quantization/utils/fp8_utils.py ->
            model_executor/.../quantization/utils/fp8_utils.py
            vllm/model_executor/layers/quantization/awq.py ->
            model_executor/layers/quantization/awq.py

            Args:
                relpath (Path): The relative path to be shortened.
            Returns:
                str: The shortened path string for display.

    This formatter adds colors by injecting them into the format string for
    static elements (timestamp, filename, line number) and modifying the
    levelname attribute for dynamic color selection.
