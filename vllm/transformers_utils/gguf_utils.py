# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from pathlib import Path
from typing import Any, Optional, Tuple
from functools import cache
from vllm.logger import init_logger

logger = init_logger(__name__)

def check_gguf_file(model: str | Path) -> bool:
    """Check if a file is a valid GGUF file."""
    model = Path(model)
    if not model.is_file():
        return False
    elif model.suffix == ".gguf":
        return True
    try:
        with model.open("rb") as f:
            header = f.read(4)
        return header == b"GGUF"
    except Exception:
        return False

@cache
def is_remote_gguf(model: str | Path) -> bool:
    """Check if the string follows repo_id:quant_type format."""
    model_str = str(model)
    return ":" in model_str and not os.path.exists(model_str)

def split_remote_gguf(model: str) -> Tuple[str, str]:
    """Split repo_id:quant_type into its components."""
    if ":" in model:
        parts = model.rsplit(":", 1)
        return parts[0], parts[1]
    return model, ""

def is_gguf(model: str | Path) -> bool:
    """Detect if the model is GGUF."""
    model_str = str(model)
    if check_gguf_file(model_str):
        return True
    return is_remote_gguf(model_str)

def detect_gguf_multimodal(model: str) -> Path | None:
    """Detect associated mmproj.gguf files."""
    return None

def maybe_patch_hf_config_from_gguf(model: str, hf_config: Any) -> Any:
    """Optional patch for multimodal GGUF configs."""
    return hf_config

def get_gguf_file_path_from_hf(repo_id: str | Path, quant_type: str, revision: str | None = "main") -> str:
    """Resolves GGUF filename in a HF repo."""
    return f"{repo_id}-{quant_type}.gguf"
