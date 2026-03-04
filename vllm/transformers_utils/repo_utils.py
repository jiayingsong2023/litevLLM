# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import json
import time
from pathlib import Path
from typing import Any, Optional, Union, List, Callable
from huggingface_hub import hf_hub_download, try_to_load_from_cache, list_repo_files as hf_list_repo_files
from vllm.logger import init_logger

logger = init_logger(__name__)

def with_retry(fn: Callable, max_retries: int = 3, delay: float = 1.0):
    """A simple retry wrapper for network operations."""
    def wrapper(*args, **kwargs):
        retries = 0
        while retries < max_retries:
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                retries += 1
                if retries >= max_retries: raise e
                time.sleep(delay * retries)
        return None
    return wrapper

def file_or_path_exists(model: str | Path) -> bool:
    return os.path.exists(str(model))

def list_repo_files(model: str | Path, revision: Optional[str] = "main") -> List[str]:
    if os.path.isdir(str(model)):
        return os.listdir(str(model))
    try:
        return hf_list_repo_files(str(model), revision=revision)
    except Exception:
        return []

def is_mistral_model_repo(model: str | Path) -> bool:
    return False

def get_hf_file_content(
    file_name: str, model: str | Path, revision: str | None = "main"
) -> bytes | None:
    file_path = try_get_local_file(model=model, file_name=file_name, revision=revision)
    if file_path is None:
        try:
            hf_hub_file = hf_hub_download(model, file_name, revision=revision)
            file_path = Path(hf_hub_file)
        except Exception:
            return None
    if file_path is not None and file_path.is_file():
        with open(file_path, "rb") as file:
            return file.read()
    return None

def try_get_local_file(
    model: str | Path, file_name: str, revision: str | None = "main"
) -> Path | None:
    file_path = Path(model) / file_name
    if file_path.is_file():
        return file_path
    else:
        try:
            cached_filepath = try_to_load_from_cache(
                repo_id=str(model), filename=file_name, revision=revision
            )
            if isinstance(cached_filepath, str):
                return Path(cached_filepath)
        except Exception:
            ...
    return None

def get_hf_file_to_dict(
    file_name: str, model: str | Path, revision: str | None = "main"
):
    file_path = try_get_local_file(model=model, file_name=file_name, revision=revision)
    if file_path is None:
        try:
            hf_hub_file = hf_hub_download(model, file_name, revision=revision)
            file_path = Path(hf_hub_file)
        except Exception:
            return None
    if file_path is not None and file_path.is_file():
        with open(file_path) as file:
            return json.load(file)
    return None
