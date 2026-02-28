# SPDX-License-Identifier: Apache-2.0
import torch
import numpy as np
import pybase64
import io

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

# Stubs for other missing functions/classes
class EmbedDType: pass
class EncodingFormat: pass
class Endianness: pass

def encode_pooling_bytes(*args, **kwargs): return b""
def encode_pooling_output(*args, **kwargs): return {}
