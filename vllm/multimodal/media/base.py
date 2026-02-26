# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np

_T = TypeVar("_T")

@dataclass
class MediaWithBytes(Generic[_T]):

    media: _T
    original_bytes: bytes

    def __array__(self, *args, **kwargs) -> np.ndarray:
        return getattr(self.media, name)

class MediaIO(ABC, Generic[_T]):
    @abstractmethod
    def load_bytes(self, data: bytes) -> _T:
        raise NotImplementedError

    @abstractmethod
    def load_base64(self, media_type: str, data: str) -> _T:
        raise NotImplementedError

    @abstractmethod
    def load_file(self, filepath: Path) -> _T:
        raise NotImplementedError
