# SPDX-License-Identifier: Apache-2.0
import os

VLLM_LOGGING_LEVEL = os.getenv("VLLM_LOGGING_LEVEL", "INFO")
VLLM_CACHE_ROOT = os.path.expanduser(os.getenv("VLLM_CACHE_ROOT", "~/.cache/vllm"))
VLLM_USE_MODELSCOPE = os.getenv("VLLM_USE_MODELSCOPE", "False").lower() == "true"
VLLM_CONFIGURE_LOGGING = os.getenv("VLLM_CONFIGURE_LOGGING", "True").lower() == "true"
VLLM_LOGGING_CONFIG_PATH = os.getenv("VLLM_LOGGING_CONFIG_PATH", None)
VLLM_LOGGING_STREAM = os.getenv("VLLM_LOGGING_STREAM", "stderr")