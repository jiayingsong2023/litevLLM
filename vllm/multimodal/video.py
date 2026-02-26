# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from abc import abstractmethod
from io import BytesIO
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import cv2

from vllm.logger import init_logger
from vllm.utils.registry import ExtensionManager

logger = init_logger(__name__)

def resize_video(frames: npt.NDArray, size: tuple[int, int]) -> npt.NDArray:
    num_frames, _, _, channels = frames.shape
    new_height, new_width = size
    resized_frames = np.empty(
        (num_frames, new_height, new_width, channels), dtype=frames.dtype
    )
    # lazy import cv2 to avoid bothering users who only use text models
    import cv2

    for i, frame in enumerate(frames):
        resized_frame = cv2.resize(frame, (new_width, new_height))
        resized_frames[i] = resized_frame
    return resized_frames

def rescale_video_size(frames: npt.NDArray, size_factor: float) -> npt.NDArray:
    _, height, width, _ = frames.shape
    new_height = int(height * size_factor)
    new_width = int(width * size_factor)

    return resize_video(frames, (new_height, new_width))

def sample_frames_from_video(frames: npt.NDArray, num_frames: int) -> npt.NDArray:
    total_frames = frames.shape[0]
    if num_frames == -1:
        return frames

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    sampled_frames = frames[frame_indices, ...]
    return sampled_frames

class VideoLoader:
    @classmethod
    @abstractmethod
    def load_bytes(
        cls, data: bytes, num_frames: int = -1, **kwargs
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        raise NotImplementedError

    @staticmethod
    def _can_use_for_recovery(
        idx: int,
        failed_frames: list[int],
        next_target_map: dict[int, int],
        total_frames: int,
    ) -> bool:
        Read frames with dynamic window forward-scan recovery.

        When a target frame fails to load, the next successfully grabbed
        frame (before the next target frame) will be used to recover it.

        Args:
            cap: OpenCV VideoCapture object
            frame_indices: Sorted list of target frame indices to load
            total_frames: Total number of frames in the video

        Returns:
            Tuple of (frames_array, valid_frame_indices, recovered_map)
            - frames_array: Array of loaded frames
            - valid_frame_indices: List of frame indices that were loaded
            - recovered_map: Dict mapping recovered_idx -> source_idx
        Load video frames from bytes.

        Args:
            data: Raw video bytes
            num_frames: Target number of frames to sample (-1 for all)
            fps: Target FPS for sampling (-1 for original)
            max_duration: Maximum duration (unused in base backend)
            frame_recovery: Enable forward-scan recovery for failed frames

        Returns:
            Tuple of (frames_array, metadata_dict)
        Load video frames with dynamic sampling based on duration.

        Args:
            data: Raw video bytes
            num_frames: Not used in dynamic backend
            fps: Target FPS for sampling (default: 2)
            max_duration: Maximum video duration to process (default: 300s)
            frame_recovery: Enable forward-scan recovery for failed frames

        Returns:
            Tuple of (frames_array, metadata_dict)
        Return the subset of `video_fps` factors that remain multiples
        of `sampling_fps`.

        Examples:
            >>> get_candidate_target_fps(video_fps=6, sampling_fps=2)
            [2, 6]
            >>> get_candidate_target_fps(video_fps=5, sampling_fps=1)
            [1, 5]
            >>> get_candidate_target_fps(video_fps=2, sampling_fps=2)
            [2]
            >>> get_candidate_target_fps(video_fps=5, sampling_fps=2)
            Traceback (most recent call last):
                ...
            ValueError: sampling_fps=2 must divide video_fps=5 to produce
                consistent frame steps.
        Get the target fps that best spans the videoand has the most frames sampled
