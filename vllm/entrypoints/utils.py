# SPDX-License-Identifier: Apache-2.0
import asyncio
import functools
import os


VLLM_SUBCMD_PARSER_EPILOG = "Use `vllm {subcmd} --help` for details."


def log_version_and_model(logger, version, model):
    logger.info(f"vLLM version {version}")
    logger.info(f"Model: {model}")


def cli_env_setup() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def get_max_tokens(request) -> int | None:
    return getattr(request, "max_completion_tokens", None) or getattr(
        request, "max_tokens", None
    )


def sanitize_message(message: str | Exception) -> str:
    return str(message).strip()


def with_cancellation(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except asyncio.CancelledError:
            raise

    return wrapper
