# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import mimetypes
import warnings
from collections.abc import Generator
from itertools import groupby
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from vllm.logger import init_logger
from vllm.utils.import_utils import LazyLoader

from .inputs import (
    BatchedTensorInputs,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    MultiModalPlaceholderDict,
)
from .media import AudioMediaIO, ImageMediaIO, MediaConnector, VideoMediaIO

if TYPE_CHECKING:
    import torch.types
else:
    torch = LazyLoader("torch", globals(), "torch")

logger = init_logger(__name__)

def __getattr__(name: str):
    if name == "MEDIA_CONNECTOR_REGISTRY":
        from .media import MEDIA_CONNECTOR_REGISTRY

        warnings.warn(
            "`vllm.multimodal.utils.MEDIA_CONNECTOR_REGISTRY` "
            "has been moved to `vllm.multimodal.media.MEDIA_CONNECTOR_REGISTRY`. "
            "The old name will be removed in v0.17.",
            DeprecationWarning,
            stacklevel=2,
        )

        return MEDIA_CONNECTOR_REGISTRY

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def encode_audio_base64(
    audio: np.ndarray,
    sampling_rate: int,
    *,
    format: str = "WAV",
) -> str:
    audio_b64 = encode_audio_base64(audio, sampling_rate, format=format)
    mimetype = mimetypes.types_map.get("." + format.lower(), "audio")
    return f"data:{mimetype};base64,{audio_b64}"

def encode_image_base64(
    image: Image.Image,
    *,
    image_mode: str = "RGB",
    format: str | None = None,
) -> str:
    image_io = ImageMediaIO(image_mode=image_mode)
    return image_io.encode_base64(image, image_format=format)

def encode_image_url(
    image: Image.Image,
    *,
    image_mode: str = "RGB",
    format: str = "PNG",
) -> str:
    image_b64 = encode_image_base64(image, image_mode=image_mode, format=format)
    mimetype = mimetypes.types_map.get("." + format.lower(), "image")
    return f"data:{mimetype};base64,{image_b64}"

def encode_video_base64(
    frames: npt.NDArray,
    *,
    format: str = "JPEG",
) -> str:
    image_io = ImageMediaIO()
    video_io = VideoMediaIO(image_io)
    return video_io.encode_base64(frames, video_format=format)

def encode_video_url(
    frames: npt.NDArray,
    *,
    format: str = "JPEG",
) -> str:
    video_b64 = encode_video_base64(frames, format=format)

    if format.lower() == "jpeg":
        mimetype = "video/jpeg"
    else:
        mimetype = mimetypes.types_map.get("." + format.lower(), "video")

    return f"data:{mimetype};base64,{video_b64}"

def argsort_mm_positions(
    mm_positions: MultiModalPlaceholderDict,
) -> list[tuple[str, int]]:
    flat_items = (
        (modality, idx, item)
        for modality, items in mm_positions.items()
        for idx, item in enumerate(items)
    )

    sorted_flat_items = sorted(flat_items, key=lambda x: x[2].offset)

    return [(modality, idx) for modality, idx, _ in sorted_flat_items]

def group_mm_kwargs_by_modality(
    mm_kwargs: list[tuple[str, MultiModalKwargsItem]],
    *,
    device: torch.types.Device = None,
    pin_memory: bool = False,
) -> Generator[tuple[str, int, BatchedTensorInputs], None, None]:
    for modality, group in groupby(mm_kwargs, key=lambda x: x[0]):
        items_lst = [item for _, item in group]
        mm_kwargs_items = MultiModalKwargsItems({modality: items_lst})
        mm_kwargs_data = mm_kwargs_items.get_data(
            device=device,
            pin_memory=pin_memory,
        )

        yield modality, len(items_lst), mm_kwargs_data

def fetch_audio(
    audio_url: str,
    audio_io_kwargs: dict[str, Any] | None = None,
) -> tuple[np.ndarray, int | float]:
    media_io_kwargs = None if not audio_io_kwargs else {"audio": audio_io_kwargs}
    media_connector = MediaConnector(
        media_io_kwargs=media_io_kwargs,
        allowed_local_media_path="/",
    )
    return media_connector.fetch_audio(audio_url)

def fetch_image(
    image_url: str,
    image_io_kwargs: dict[str, Any] | None = None,
) -> Image.Image:
    media_io_kwargs = None if not image_io_kwargs else {"image": image_io_kwargs}
    media_connector = MediaConnector(
        media_io_kwargs=media_io_kwargs,
        allowed_local_media_path="/",
    )
    return media_connector.fetch_image(image_url)

def fetch_video(
    video_url: str,
    video_io_kwargs: dict[str, Any] | None = None,
) -> tuple[npt.NDArray, dict[str, Any]]:
    media_io_kwargs = None if not video_io_kwargs else {"video": video_io_kwargs}
    media_connector = MediaConnector(
        media_io_kwargs=media_io_kwargs,
        allowed_local_media_path="/",
    )
    return media_connector.fetch_video(video_url)
