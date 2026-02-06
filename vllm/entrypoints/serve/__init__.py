# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import FastAPI


def register_vllm_serve_api_routers(app: FastAPI):
    """Deprecated entrypoint. Keep tokenization only for compatibility."""
    from vllm.entrypoints.serve.tokenize.api_router import (
        attach_router as attach_tokenize_router,
    )

    attach_tokenize_router(app)
