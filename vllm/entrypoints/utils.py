# SPDX-License-Identifier: Apache-2.0
def log_version_and_model(logger, version, model):
    logger.info(f"vLLM version {version}")
    logger.info(f"Model: {model}")
