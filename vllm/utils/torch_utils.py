# SPDX-License-Identifier: Apache-2.0
import torch

def get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
