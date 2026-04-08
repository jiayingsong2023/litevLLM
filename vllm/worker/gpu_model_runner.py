# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim for the legacy ``vllm.worker.gpu_model_runner`` import path.

Lite-only runtime work should use ``vllm.engine.lite_engine.LiteEngine`` as the
official execution path. This module remains only to avoid breaking internal
imports that still reference the old location.
"""

from vllm.worker.gpu.model_runner import GPUModelRunner

__all__ = ["GPUModelRunner"]
