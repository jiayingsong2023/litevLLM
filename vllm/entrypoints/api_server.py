# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility shim for legacy api_server entrypoint.

This module keeps legacy import paths stable while delegating all serving
behavior to the Lite OpenAI-compatible API server implementation.
"""

from vllm.logger import init_logger

logger = init_logger(__name__)

# Re-export FastAPI app for callers importing from the legacy module.
from vllm.entrypoints.openai.api_server import app  # noqa: E402
from vllm.entrypoints.openai.api_server import main as _openai_main  # noqa: E402


def main() -> None:
    # Keep warning concise but explicit for migration.
    logger.warning(
        "vllm.entrypoints.api_server is deprecated; "
        "please use vllm.entrypoints.openai.api_server."
    )
    _openai_main()


if __name__ == "__main__":
    main()
