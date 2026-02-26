# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
from collections.abc import Sequence
from concurrent.futures.process import ProcessPoolExecutor
from functools import cache
from typing import Any

import torch

def cuda_is_initialized() -> bool:
    if not torch.xpu._is_compiled():
        return False
    return torch.xpu.is_initialized()

def get_cu_count(device_id: int = 0) -> int:
    # UVA requires pinned memory.
    # TODO: Add more requirements for UVA if needed.
    return is_pin_memory_available()
