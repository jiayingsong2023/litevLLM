# SPDX-License-Identifier: Apache-2.0
import logging
import os
from typing import Any

def init_logger(name: str):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s',
            datefmt='%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(os.getenv("VLLM_LOGGING_LEVEL", "INFO"))
        logger.propagate = False
    return logger