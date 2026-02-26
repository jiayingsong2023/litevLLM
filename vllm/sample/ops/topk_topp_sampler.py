# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn
from packaging import version

from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config.model import LogprobsMode
from vllm.logger import init_logger
from vllm.platforms import CpuArchEnum, current_platform

logger = init_logger(__name__)

class TopKTopPSampler(nn.Module):

    def __init__(self, logprobs_mode: LogprobsMode = "raw_logprobs") -> None:
        super().__init__()
        self.logprobs_mode = logprobs_mode
        
        # Simplification: Use forward_native for GPU (ROCm/CUDA) and fallback
        if current_platform.is_cuda() or current_platform.is_rocm():
            logger.info("Simplification: Forcing forward_native for TopKTopPSampler")
            self.forward = self.forward_native
        elif current_platform.is_cpu():
            arch = current_platform.get_cpu_architecture()
            if arch in (CpuArchEnum.RISCV, CpuArchEnum.POWERPC):
                self.forward = self.forward_native
            else:
                self.forward = self.forward_cpu
        else:
            self.forward = self.forward_native

        self.apply_top_k_top_p = apply_top_k_top_p

    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.apply_top_k_top_p(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return random_sample(probs, generators), logits_to_return

    def forward_cuda(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        PyTorch-native implementation of top-k and top-p sampling for CPU.

        The logits tensor may be updated in-place.
        if (k is None and p is None) or generators:
            if generators:
                logger.warning_once(
                    "aiter sampler does not support per-request generators; "
                    "falling back to PyTorch-native."
                )
            return self.forward_native(logits, generators, k, p)
        assert self.logprobs_mode not in (
            "processed_logits",
            "processed_logprobs",
        ), "aiter sampler does not support returning logits/logprobs."
        if DISABLE_AITER_SAMPLER:
            return self.forward_native(logits, generators, k, p)
        return self.aiter_sample(logits, k, p, generators), None

    def aiter_sample(
        self,
        logits: torch.Tensor,
        k: torch.Tensor | None,
        p: torch.Tensor | None,
        generators: dict[int, torch.Generator],
    ) -> torch.Tensor:

    If a top-p is used, this function will sort the logits tensor,
    which can be slow for large batches.

    The logits tensor may be updated in-place.
    Apply top-k mask to the logits.

    This implementation doesn't involve sorting the entire vocab.

    The logits tensor may be updated in-place.

    We use this function instead of torch.multinomial because torch.multinomial
    causes CPU-GPU synchronization.

    Statistically, this function is equivalent to the `random_sample` function.
    However, this function is faster because it avoids sorting the logits tensor
    via rejection sampling.

    NOTE: The outputs of this function do not necessarily match the outputs of
    the `random_sample` function. It only guarantees that the outputs are
    statistically equivalent.

    NOTE: This function includes CPU-GPU synchronization, while `random_sample`
    does not. Call this function at the end of the forward pass to minimize
    the synchronization overhead.
