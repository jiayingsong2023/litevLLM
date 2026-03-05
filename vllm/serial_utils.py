# SPDX-License-Identifier: Apache-2.0
import torch
import numpy as np
import pybase64
import io
from typing import Literal, TypeAlias

def tensor2base64(tensor: torch.Tensor) -> str:
    """Serialize a torch tensor to a base64 string."""
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return pybase64.b64encode(buffer.getvalue()).decode("utf-8")

def base642tensor(b64_str: str) -> torch.Tensor:
    """Deserialize a base64 string to a torch tensor."""
    data = pybase64.b64decode(b64_str)
    buffer = io.BytesIO(data)
    return torch.load(buffer)

# Minimal type aliases for serving schemas.
EmbedDType: TypeAlias = Literal["float32", "float16", "bfloat16"]
EncodingFormat: TypeAlias = Literal["float", "base64", "bytes", "bytes_only"]
Endianness: TypeAlias = Literal["little", "big"]


def encode_pooling_bytes(*args, **kwargs):
    del args, kwargs
    return b"", [], {"prompt_tokens": 0, "total_tokens": 0}


def encode_pooling_output(*args, **kwargs):
    del args, kwargs
    return []
