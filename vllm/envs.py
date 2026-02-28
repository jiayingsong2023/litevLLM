# SPDX-License-Identifier: Apache-2.0
import os

VLLM_LOGGING_LEVEL = os.getenv("VLLM_LOGGING_LEVEL", "INFO")
VLLM_CACHE_ROOT = os.path.expanduser(os.getenv("VLLM_CACHE_ROOT", "~/.cache/vllm"))
VLLM_USE_MODELSCOPE = os.getenv("VLLM_USE_MODELSCOPE", "False").lower() == "true"
VLLM_CONFIGURE_LOGGING = os.getenv("VLLM_CONFIGURE_LOGGING", "True").lower() == "true"
VLLM_LOGGING_CONFIG_PATH = os.getenv("VLLM_LOGGING_CONFIG_PATH", None)
VLLM_LOGGING_STREAM = os.getenv("VLLM_LOGGING_STREAM", "stderr")
VLLM_MEDIA_LOADING_THREAD_COUNT = int(os.getenv("VLLM_MEDIA_LOADING_THREAD_COUNT", "8"))
VLLM_USE_V2_MODEL_RUNNER = os.getenv("VLLM_USE_V2_MODEL_RUNNER", "False").lower() == "true"
VLLM_INSTANCE_ID = os.getenv("VLLM_INSTANCE_ID", "default")
