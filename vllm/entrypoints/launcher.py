# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import signal
import socket
from http import HTTPStatus
from typing import Any

import uvicorn
from fastapi import FastAPI, Request, Response

from vllm import envs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.constants import (
    H11_MAX_HEADER_COUNT_DEFAULT,
    H11_MAX_INCOMPLETE_EVENT_SIZE_DEFAULT,
)
from vllm.entrypoints.ssl import SSLCertRefresher
from vllm.logger import init_logger
from vllm.utils.network_utils import find_process_using_port
from vllm.engine.v1.exceptions import EngineDeadError, EngineGenerateError

logger = init_logger(__name__)

async def serve_http(
    app: FastAPI,
    sock: socket.socket | None,
    enable_ssl_refresh: bool = False,
    **uvicorn_kwargs: Any,
):
    logger.info("Available routes are:")
    # post endpoints
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)

        if methods is None or path is None:
            continue

        logger.info("Route: %s, Methods: %s", path, ", ".join(methods))

    # other endpoints
    for route in app.routes:
        endpoint = getattr(route, "endpoint", None)
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)

        if endpoint is None or path is None or methods is not None:
            continue

        logger.info("Route: %s, Endpoint: %s", path, endpoint.__name__)

    # Extract header limit options if present
    h11_max_incomplete_event_size = uvicorn_kwargs.pop(
        "h11_max_incomplete_event_size", None
    )
    h11_max_header_count = uvicorn_kwargs.pop("h11_max_header_count", None)

    # Set safe defaults if not provided
    if h11_max_incomplete_event_size is None:
        h11_max_incomplete_event_size = H11_MAX_INCOMPLETE_EVENT_SIZE_DEFAULT
    if h11_max_header_count is None:
        h11_max_header_count = H11_MAX_HEADER_COUNT_DEFAULT

    config = uvicorn.Config(app, **uvicorn_kwargs)
    # Set header limits
    config.h11_max_incomplete_event_size = h11_max_incomplete_event_size
    config.h11_max_header_count = h11_max_header_count
    config.load()
    server = uvicorn.Server(config)
    _add_shutdown_handlers(app, server)

    loop = asyncio.get_running_loop()

    watchdog_task = loop.create_task(watchdog_loop(server, app.state.engine_client))
    server_task = loop.create_task(server.serve(sockets=[sock] if sock else None))

    ssl_cert_refresher = (
        None
        if not enable_ssl_refresh
        else SSLCertRefresher(
            ssl_context=config.ssl,
            key_path=config.ssl_keyfile,
            cert_path=config.ssl_certfile,
            ca_path=config.ssl_ca_certs,
        )
    )

    def signal_handler() -> None:
        # prevents the uvicorn signal handler to exit early
        server_task.cancel()
        watchdog_task.cancel()
        if ssl_cert_refresher:
            ssl_cert_refresher.stop()

    async def dummy_shutdown() -> None:
        pass

    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    try:
        await server_task
        return dummy_shutdown()
    except asyncio.CancelledError:
        port = uvicorn_kwargs["port"]
        process = find_process_using_port(port)
        if process is not None:
            logger.warning(
                "port %s is used by process %s launched with command:\n%s",
                port,
                process,
                " ".join(process.cmdline()),
            )
        logger.info("Shutting down FastAPI HTTP server.")
        return server.shutdown()
    finally:
        watchdog_task.cancel()

async def watchdog_loop(server: uvicorn.Server, engine: EngineClient):
    VLLM_WATCHDOG_TIME_S = 5.0
    while True:
        await asyncio.sleep(VLLM_WATCHDOG_TIME_S)
        terminate_if_errored(server, engine)

def terminate_if_errored(server: uvicorn.Server, engine: EngineClient):
    engine_errored = engine.errored and not engine.is_running
    if not envs.VLLM_KEEP_ALIVE_ON_ENGINE_DEATH and engine_errored:
        server.should_exit = True

def _add_shutdown_handlers(app: FastAPI, server: uvicorn.Server) -> None:

    @app.exception_handler(RuntimeError)
    @app.exception_handler(EngineDeadError)
    @app.exception_handler(EngineGenerateError)
    async def runtime_exception_handler(request: Request, __):
        terminate_if_errored(
            server=server,
            engine=request.app.state.engine_client,
        )

        return Response(status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
