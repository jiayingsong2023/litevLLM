# SPDX-License-Identifier: Apache-2.0
import os

VLLM_SUBCMD_PARSER_EPILOG = "Use `vllm {subcmd} --help` for details."


def log_version_and_model(logger, version, model):
    logger.info("vLLM version %s", version)
    logger.info("Model: %s", model)


def cli_env_setup() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def get_max_tokens(request) -> int | None:
    return getattr(request, "max_completion_tokens", None) or getattr(
        request, "max_tokens", None
    )


def sanitize_message(message: str | Exception) -> str:
    return str(message).strip()
