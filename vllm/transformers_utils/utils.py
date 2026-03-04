# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import struct
from functools import cache
from os import PathLike
from pathlib import Path
from typing import Any, Dict

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
    """List files in a modelscope repo (Shim)."""
    return []

def parse_safetensors_file_metadata(path: str | PathLike) -> Dict[str, str]:
    """Parses the metadata from a safetensors file."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_size).decode("utf-8")
        header = json.loads(header_json)
        return header.get("__metadata__", {})
