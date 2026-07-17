# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility shim for legacy api_server entrypoint.

This module keeps legacy import paths stable while delegating all serving
behavior to the Lite OpenAI-compatible API server implementation.
"""

from vllm.entrypoints.openai.api_server import app as app
from vllm.entrypoints.openai.api_server import main as _openai_main
from vllm.logger import init_logger

logger = init_logger(__name__)


def main() -> None:
    # Keep warning concise but explicit for migration.
    logger.warning(
        "vllm.entrypoints.api_server is deprecated; "
        "please use vllm.entrypoints.openai.api_server."
    )
    _openai_main()


if __name__ == "__main__":
    main()
