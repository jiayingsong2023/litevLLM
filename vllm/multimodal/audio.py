# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
import numpy.typing as npt
import torch

from vllm.utils.import_utils import PlaceholderModule

try:
    import librosa
except ImportError:
    librosa = PlaceholderModule("librosa")  # type: ignore[assignment]

try:
    import scipy.signal as scipy_signal
except ImportError:
    scipy_signal = PlaceholderModule("scipy").placeholder_attr("signal")  # type: ignore[assignment]

# ============================================================

class ChannelReduction(str, Enum):

    This dataclass defines the expected audio format for a model's feature
    extractor. It is used to normalize audio data before processing.

    Attributes:
        target_channels: Number of output channels. None means passthrough
            (no normalization). 1 = mono, 2 = stereo, etc.
        channel_reduction: Method to reduce channels when input has more
            channels than target. Only used when reducing channels.
        return self.target_channels is not None

    def __repr__(self) -> str:
        if self.target_channels is None:
            return "AudioSpec(passthrough)"
        return (
            f"AudioSpec(channels={self.target_channels}, "
            f"reduction={self.channel_reduction.value})"
        )

# Pre-defined specs for common use cases
MONO_AUDIO_SPEC = AudioSpec(target_channels=1, channel_reduction=ChannelReduction.MEAN)
PASSTHROUGH_AUDIO_SPEC = AudioSpec(target_channels=None)

def normalize_audio(
    audio: npt.NDArray[np.floating] | torch.Tensor,
    spec: AudioSpec,
) -> npt.NDArray[np.floating] | torch.Tensor:
    if not spec.needs_normalization:
        return audio

    # Handle 1D audio (already mono)
    if audio.ndim == 1:
        if spec.target_channels == 1:
            return audio
        raise ValueError(f"Cannot expand mono audio to {spec.target_channels} channels")

    # Handle 2D audio
    if audio.ndim != 2:
        raise ValueError(f"Unsupported audio shape: {audio.shape}. Expected 1D or 2D.")

    # Auto-detect format: if shape[0] > shape[1], assume (time, channels)
    # This handles soundfile format where time dimension is typically much larger
    if audio.shape[0] > audio.shape[1]:
        # Transpose from (time, channels) to (channels, time)
        audio = audio.T if isinstance(audio, np.ndarray) else audio.T

    num_channels = audio.shape[0]

    # No reduction needed if already at target
    if num_channels == spec.target_channels:
        return audio

    # Cannot expand channels
    if num_channels < spec.target_channels:
        raise ValueError(
            f"Cannot expand {num_channels} channels to {spec.target_channels}"
        )

    # Reduce channels
    is_numpy = isinstance(audio, np.ndarray)

    if spec.target_channels == 1:
        # Reduce to mono
        if spec.channel_reduction == ChannelReduction.MEAN:
            result = np.mean(audio, axis=0) if is_numpy else audio.mean(dim=0)
        elif spec.channel_reduction == ChannelReduction.FIRST:
            result = audio[0]
        elif spec.channel_reduction == ChannelReduction.MAX:
            result = np.max(audio, axis=0) if is_numpy else audio.max(dim=0).values
        elif spec.channel_reduction == ChannelReduction.SUM:
            result = np.sum(audio, axis=0) if is_numpy else audio.sum(dim=0)
        else:
            raise ValueError(f"Unknown reduction method: {spec.channel_reduction}")
        return result
    else:
        # Reduce to N channels (take first N and apply reduction if needed)
        # For now, just take first N channels
        return audio[: spec.target_channels]

# ============================================================
# Audio Resampling
# ============================================================

def resample_audio_librosa(
    audio: npt.NDArray[np.floating],
    *,
    orig_sr: float,
    target_sr: float,
) -> npt.NDArray[np.floating]:
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

def resample_audio_scipy(
    audio: npt.NDArray[np.floating],
    *,
    orig_sr: float,
    target_sr: float,
):
    if orig_sr > target_sr:
        return scipy_signal.resample_poly(audio, 1, orig_sr // target_sr)
    elif orig_sr < target_sr:
        return scipy_signal.resample_poly(audio, target_sr // orig_sr, 1)
    return audio

class AudioResampler:
