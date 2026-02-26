# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from urllib.parse import urlparse

class UvicornAccessLogFilter(logging.Filter):

    def __init__(self, excluded_paths: list[str] | None = None):
        super().__init__()
        self.excluded_paths = set(excluded_paths or [])

    def filter(self, record: logging.LogRecord) -> bool:
        if not self.excluded_paths:
            return True

        # This filter is specific to uvicorn's access logs.
        if record.name != "uvicorn.access":
            return True

        # The path is the 3rd argument in the log record's args tuple.
        # See uvicorn's access logging implementation for details.
        log_args = record.args
        if isinstance(log_args, tuple) and len(log_args) >= 3:
            path_with_query = log_args[2]
            # Get path component without query string.
            if isinstance(path_with_query, str):
                path = urlparse(path_with_query).path
                if path in self.excluded_paths:
                    return False

        return True

def create_uvicorn_log_config(
    excluded_paths: list[str] | None = None,
    log_level: str = "info",
) -> dict:
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "access_log_filter": {
                "()": UvicornAccessLogFilter,
                "excluded_paths": excluded_paths or [],
            },
        },
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',  # noqa: E501
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "filters": ["access_log_filter"],
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["default"],
                "level": log_level.upper(),
                "propagate": False,
            },
            "uvicorn.error": {
                "level": log_level.upper(),
                "handlers": ["default"],
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["access"],
                "level": log_level.upper(),
                "propagate": False,
            },
        },
    }
    return config
