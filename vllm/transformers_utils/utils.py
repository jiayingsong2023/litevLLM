# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import struct
from functools import cache
from os import PathLike
from pathlib import Path
from typing import Any

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

def is_s3(model_or_path: str) -> bool:
    return model_or_path.lower().startswith("s3://")

def is_gcs(model_or_path: str) -> bool:
    return model_or_path.lower().startswith("gs://")

def is_cloud_storage(model_or_path: str) -> bool:
    return is_s3(model_or_path) or is_gcs(model_or_path)

def modelscope_list_repo_files(
    repo_id: str,
    revision: str | None = None,
    token: str | bool | None = None,
) -> list[str]:
    Use model_redirect to redirect the model name to a local folder.

    :param model: hf model name
    :return: maybe redirect to a local folder
