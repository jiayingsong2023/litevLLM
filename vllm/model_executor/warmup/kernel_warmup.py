# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import TYPE_CHECKING, Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.utils.flashinfer import has_flashinfer

if TYPE_CHECKING:
    from vllm.worker.gpu_model_runner import GPUModelRunner
    from vllm.worker.gpu_worker import Worker

logger = init_logger(__name__)


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return max(minimum, int(default))
    try:
        return max(minimum, int(raw))
    except ValueError:
        logger.warning("Invalid %s=%r, fallback to %d", name, raw, default)
        return max(minimum, int(default))


def _run_first_request_warmup(worker: "Worker") -> None:
    runner: Any = worker.model_runner
    steps = _env_int("FASTINFERENCE_WARMUP_FIRST_REQUEST_STEPS", 0, minimum=0)
    if steps <= 0:
        return
    if not hasattr(runner, "_dummy_run"):
        logger.warning("First-request warmup skipped: model_runner has no _dummy_run")
        return

    max_tokens = int(getattr(runner.scheduler_config, "max_num_batched_tokens", 128))
    prefill_tokens = _env_int(
        "FASTINFERENCE_WARMUP_PREFILL_TOKENS",
        min(max_tokens, 1024),
        minimum=1,
    )
    decode_tokens = _env_int(
        "FASTINFERENCE_WARMUP_DECODE_TOKENS",
        min(max_tokens, 16),
        minimum=1,
    )
    prefill_tokens = min(prefill_tokens, max_tokens)
    decode_tokens = min(decode_tokens, max_tokens)
    warmup_sync = _env_truthy("FASTINFERENCE_WARMUP_SYNC", default=True)

    logger.info(
        "Running first-request warmup: steps=%d prefill_tokens=%d decode_tokens=%d",
        steps,
        prefill_tokens,
        decode_tokens,
    )
    for idx in range(steps):
        # Step 0: prefill-shape compile; later steps: decode-ish short path compile.
        num_tokens = prefill_tokens if idx == 0 else decode_tokens
        runner._dummy_run(
            num_tokens,
            skip_attn=False,
            skip_eplb=True,
            is_profile=True,
        )

    if warmup_sync:
        torch.cuda.synchronize()


def kernel_warmup(worker: "Worker"):
    # Deep GEMM warmup
    do_deep_gemm_warmup = (
        bool(getattr(envs, "VLLM_USE_DEEP_GEMM", False))
        and is_deep_gemm_supported()
        and str(getattr(envs, "VLLM_DEEP_GEMM_WARMUP", "skip")).lower() != "skip"
    )
    if do_deep_gemm_warmup:
        try:
            from vllm.model_executor.warmup.deep_gemm_warmup import deep_gemm_warmup
        except Exception as exc:
            logger.warning(
                "Deep GEMM warmup skipped: %s: %s",
                type(exc).__name__,
                exc,
            )
        else:
            model = worker.get_model()
            max_tokens = worker.scheduler_config.max_num_batched_tokens
            deep_gemm_warmup(model, max_tokens)

    # FlashInfer autotune for Hopper (SM 9.0) and Blackwell (SM 10.0) GPUs
    if has_flashinfer() and current_platform.has_device_capability(90):
        flashinfer_autotune(worker.model_runner)

    # FlashInfer attention warmup
    # Only warmup if the model has FlashInfer attention groups
    # and is not a pooling model
    def _is_flashinfer_backend(backend):
        try:
            return backend.get_name() == "FLASHINFER"
        except NotImplementedError:
            return False

    if (
        not worker.model_runner.is_pooling_model
        and worker.model_runner.attn_groups
        # NOTE: This should be `any` instead of `all` but other hybrid attention
        # backends don't support this dummy run. Once we remove
        # `build_for_cudagraph_capture`, we can change it to `any`.
        and all(
            _is_flashinfer_backend(group.backend)
            for groups in worker.model_runner.attn_groups
            for group in groups
        )
    ):
        logger.info("Warming up FlashInfer attention.")
        # Warmup with mixed batch containing both prefill and decode tokens
        # This is to warm up both prefill and decode attention kernels
        worker.model_runner._dummy_run(
            num_tokens=16,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_mixed_batch=True,
        )

    _run_first_request_warmup(worker)


def flashinfer_autotune(runner: "GPUModelRunner") -> None:
    from vllm.utils.flashinfer import autotune

    with torch.inference_mode(), autotune():
        # We skip EPLB here since we don't want to record dummy metrics
        # When autotuning with number of tokens m, flashinfer will autotune
        # operations for all number of tokens up to m.
        # So we only need to run with the max number of tokens.
        runner._dummy_run(
            runner.scheduler_config.max_num_batched_tokens,
            skip_eplb=True,
            is_profile=True,
        )
