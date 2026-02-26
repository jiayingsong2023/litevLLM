# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import lru_cache
from pathlib import Path

import vllm.envs as envs
from vllm.connections import global_http_connection

VLLM_S3_BUCKET_URL = "https://vllm-public-assets.s3.us-west-2.amazonaws.com"

def get_cache_dir() -> Path:
    Download an asset file from `s3://vllm-public-assets`
    and return the path to the downloaded file.
