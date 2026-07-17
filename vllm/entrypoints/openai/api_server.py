# SPDX-License-Identifier: Apache-2.0
import argparse
import asyncio
import json
import os
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.engine.async_llm import AsyncLLM
from vllm.engine.errors import RequestRejectedError
from vllm.engine.runtime_observer import InMemoryRuntimeObserver
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.serving.config_builder import build_vllm_config

logger = init_logger(__name__)
engine: AsyncLLM | None = None
debug_endpoints_enabled = False
MAX_REQUEST_BODY_BYTES = 1 << 20


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    try:
        yield
    finally:
        if engine is not None:
            await asyncio.to_thread(engine.shutdown)


app = FastAPI(lifespan=_lifespan)


def _require_engine() -> AsyncLLM:
    if engine is None:
        raise HTTPException(status_code=503, detail="engine not initialized")
    return engine


def _require_debug_endpoints() -> None:
    if not debug_endpoints_enabled:
        raise HTTPException(status_code=404, detail="not found")


async def _read_json_object(request: Request) -> dict:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_REQUEST_BODY_BYTES:
                raise HTTPException(status_code=413, detail="request body too large")
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid Content-Length") from None

    chunks: list[bytes] = []
    received = 0
    async for chunk in request.stream():
        received += len(chunk)
        if received > MAX_REQUEST_BODY_BYTES:
            raise HTTPException(status_code=413, detail="request body too large")
        chunks.append(chunk)
    try:
        body = json.loads(b"".join(chunks))
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="request body must be valid JSON") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="request body must be a JSON object")
    return body


def _parse_structured_outputs(body: dict) -> StructuredOutputsParams | None:
    raw_structured = body.get("structured_outputs")
    if isinstance(raw_structured, dict):
        return StructuredOutputsParams(
            json=raw_structured.get("json"),
            regex=raw_structured.get("regex"),
            grammar=raw_structured.get("grammar"),
            structural_tag=raw_structured.get("structural_tag"),
            json_object=bool(raw_structured.get("json_object", False)),
            choice=list(raw_structured["choice"])
            if raw_structured.get("choice") is not None
            else None,
        )

    response_format = body.get("response_format")
    if not isinstance(response_format, dict):
        return None

    rf_type = response_format.get("type")
    if rf_type == "json_object":
        return StructuredOutputsParams(json_object=True)
    if rf_type == "json_schema":
        json_schema = response_format.get("json_schema")
        if not isinstance(json_schema, dict):
            raise HTTPException(
                status_code=400, detail="response_format.json_schema must be an object"
            )
        schema = json_schema.get("schema") or json_schema.get("json_schema")
        if not isinstance(schema, dict):
            raise HTTPException(
                status_code=400,
                detail="response_format.json_schema.schema must be an object",
            )
        return StructuredOutputsParams(json=schema)
    if rf_type == "text" or rf_type is None:
        return None
    raise HTTPException(
        status_code=400, detail=f"unsupported response_format.type: {rf_type}"
    )


def _normalize_chat_messages(messages: list[dict]) -> tuple[list[dict[str, str]], dict | None]:
    normalized: list[dict[str, str]] = []
    images: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise HTTPException(
                status_code=400, detail="each message must be an object"
            )
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise HTTPException(status_code=400, detail="message role must be a string")
        content = message.get("content")
        if isinstance(content, str):
            normalized.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            raise HTTPException(
                status_code=400,
                detail="message content must be a string or content block list",
            )
        text_parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                raise HTTPException(
                    status_code=400, detail="message content block must be an object"
                )
            block_type = item.get("type")
            if block_type == "text":
                text = item.get("text")
                if not isinstance(text, str):
                    raise HTTPException(
                        status_code=400,
                        detail="text content block must include string text",
                    )
                text_parts.append(text)
                continue
            if block_type == "image_url":
                raw = item.get("image_url")
                url = raw.get("url") if isinstance(raw, dict) else raw
                if not isinstance(url, str) or not url.strip():
                    raise HTTPException(
                        status_code=400,
                        detail="image_url block must include non-empty url",
                    )
                images.append({"image": url.strip()})
                text_parts.append("<image>")
                continue
            raise HTTPException(
                status_code=400,
                detail=f"unsupported message content type: {block_type}",
            )
        normalized.append({"role": role, "content": "\n".join(text_parts)})
    multi_modal_data = {"image": images} if images else None
    return normalized, multi_modal_data


def _parse_chat_message_content(messages: list[dict]) -> tuple[str, dict | None]:
    normalized, multi_modal_data = _normalize_chat_messages(messages)
    return (normalized[-1]["content"] if normalized else ""), multi_modal_data


def _render_chat_prompt(runtime_engine: AsyncLLM, messages: list[dict[str, str]]) -> str:
    tokenizer = getattr(getattr(runtime_engine, "engine", None), "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return "\n".join(message["content"] for message in messages)
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"chat template failed: {exc}") from exc
    if not isinstance(prompt, str):
        raise HTTPException(status_code=400, detail="chat template must return text")
    return prompt


def _sampling_params_from_body(body: dict[str, Any]) -> SamplingParams:
    if body.get("n", 1) != 1:
        raise HTTPException(status_code=400, detail="only n=1 is supported")
    if body.get("stop") not in (None, [], ""):
        raise HTTPException(status_code=400, detail="stop strings are not supported")
    if body.get("logprobs") not in (None, False, 0):
        raise HTTPException(status_code=400, detail="logprobs are not supported")
    if body.get("seed") is not None:
        raise HTTPException(status_code=400, detail="seed is not supported")
    max_tokens = body.get("max_tokens", body.get("max_completion_tokens", 128))
    return SamplingParams(
        max_tokens=max_tokens,
        temperature=body.get("temperature", 0.7),
        top_p=body.get("top_p", 1.0),
        top_k=body.get("top_k", -1),
        min_p=body.get("min_p", 0.0),
        frequency_penalty=body.get("frequency_penalty", 0.0),
        presence_penalty=body.get("presence_penalty", 0.0),
        repetition_penalty=body.get("repetition_penalty", 1.0),
        structured_outputs=_parse_structured_outputs(body),
    )


def _finish_reason(output: Any, max_tokens: int | None) -> str:
    if (
        output.outputs
        and max_tokens is not None
        and len(output.outputs[0].token_ids) >= max_tokens
    ):
        return "length"
    return "stop"


def _usage(output: Any) -> dict[str, int]:
    completion_tokens = len(output.outputs[0].token_ids) if output.outputs else 0
    prompt_tokens = len(output.prompt_token_ids)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _new_chat_request_id() -> str:
    return f"chat-{uuid.uuid4().hex}"


def _summarize_runtime_stats(stats: dict) -> dict:
    if not isinstance(stats, dict):
        return {}
    observer = stats.get("observer", {})
    lora = stats.get("lora", {})
    if not isinstance(observer, dict):
        observer = {}
    if not isinstance(lora, dict):
        lora = {}
    prefix_cache = observer.get("prefix_cache", {})
    async_driver = stats.get("async_driver", {})
    if not isinstance(prefix_cache, dict):
        prefix_cache = {}
    if not isinstance(async_driver, dict):
        async_driver = {}
    return {
        "prefix_cache": {
            "hit_rate": float(prefix_cache.get("hit_rate", 0.0) or 0.0),
            "avg_saved_prefill_tokens_per_request": float(
                prefix_cache.get("avg_saved_prefill_tokens_per_request", 0.0) or 0.0
            ),
        },
        "lora": {
            "registered_adapters": int(lora.get("registered_adapters", 0) or 0),
            "active_adapters": int(lora.get("active_adapters", 0) or 0),
            "total_routed_requests": int(lora.get("total_routed_requests", 0) or 0),
        },
        "async_driver": {
            "steps": int(async_driver.get("steps", 0) or 0),
            "backpressure_sleeps": int(async_driver.get("backpressure_sleeps", 0) or 0),
            "idle_waits": int(async_driver.get("idle_waits", 0) or 0),
            "background_errors": int(async_driver.get("background_errors", 0) or 0),
            "min_step_interval_s": float(
                async_driver.get("min_step_interval_s", 0.0) or 0.0
            ),
        },
    }


@app.get("/v1/models")
async def show_available_models():
    model_config = await _require_engine().get_model_config()
    return JSONResponse(
        content={
            "object": "list",
            "data": [
                {
                    "id": model_config.model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "fastinference",
                }
            ],
        }
    )


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    runtime_engine = _require_engine()
    if not getattr(runtime_engine, "is_healthy", lambda: True)():
        raise HTTPException(status_code=503, detail="engine is fatal")
    return {"status": "ready"}


@app.get("/debug/runtime_stats")
async def get_runtime_stats():
    _require_debug_endpoints()
    runtime_engine = _require_engine()
    model_config = await runtime_engine.get_model_config()
    stats = runtime_engine.stats()
    return JSONResponse(
        content={
            "model": getattr(model_config, "model", None),
            "summary": _summarize_runtime_stats(stats),
            "stats": stats,
        }
    )


@app.post("/debug/runtime_stats/reset")
async def reset_runtime_stats(clear_prefix_cache: bool = False):
    _require_debug_endpoints()
    runtime_engine = _require_engine()
    runtime_engine.reset_stats(clear_prefix_cache=clear_prefix_cache)
    model_config = await runtime_engine.get_model_config()
    stats = runtime_engine.stats()
    return JSONResponse(
        content={
            "model": getattr(model_config, "model", None),
            "summary": _summarize_runtime_stats(stats),
            "stats": stats,
        }
    )


@app.post("/v1/chat/completions")
async def create_chat_completion(request: Request):
    runtime_engine = _require_engine()
    body = await _read_json_object(request)
    model_name = body.get("model")
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="messages must be a list")
    model_config = await runtime_engine.get_model_config()
    if not isinstance(model_name, str) or model_name != model_config.model:
        raise HTTPException(status_code=400, detail="model does not match loaded model")

    normalized_messages, multi_modal_data = _normalize_chat_messages(messages)
    prompt = _render_chat_prompt(runtime_engine, normalized_messages)
    sampling_params = _sampling_params_from_body(body)
    request_id = _new_chat_request_id()
    try:
        await runtime_engine.submit(
            prompt,
            sampling_params,
            request_id,
            multi_modal_data=multi_modal_data,
        )
    except RequestRejectedError as exc:
        status_code = 503 if "fatal" in str(exc).lower() else 429
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def generate_stream() -> AsyncGenerator[str, None]:
        last_text = ""
        try:
            async for output in runtime_engine.stream(request_id):
                if await request.is_disconnected():
                    break
                current_text = output.outputs[0].text if output.outputs else ""
                delta_text = (
                    current_text[len(last_text) :]
                    if current_text.startswith(last_text)
                    else current_text
                )
                last_text = current_text
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": delta_text},
                            "finish_reason": (
                                _finish_reason(output, sampling_params.max_tokens)
                                if output.finished
                                else None
                            ),
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            await runtime_engine.abort(request_id)

    if stream:
        return StreamingResponse(generate_stream(), media_type="text/event-stream")
    else:
        final_output = None
        async for output in runtime_engine.stream(request_id):
            final_output = output
        if final_output is None:
            raise HTTPException(status_code=503, detail="generation ended without output")

        return JSONResponse(
            content={
                "id": request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": final_output.outputs[0].text,
                        },
                        "finish_reason": _finish_reason(
                            final_output,
                            sampling_params.max_tokens,
                        ),
                    }
                ],
                "usage": _usage(final_output),
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="FastInference OpenAI API Server")
    parser.add_argument("--model", type=str, required=True, help="Path to the model.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--enable-debug-endpoints",
        action="store_true",
        help="Enable unauthenticated /debug/* endpoints. Keep disabled on public binds.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow model-provided Python only with --revision.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Pinned model revision required with --trust-remote-code.",
    )
    parser.add_argument(
        "--policy-mode",
        type=str,
        choices=["auto", "aggressive", "stable"],
        default=os.getenv("FASTINF_POLICY_MODE", "auto"),
        help="Load-time execution policy selection mode.",
    )
    args = parser.parse_args()

    global debug_endpoints_enabled, engine
    debug_endpoints_enabled = args.enable_debug_endpoints
    v_config = build_vllm_config(
        args.model,
        policy_mode=args.policy_mode,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
    )
    object.__setattr__(v_config, "runtime_observer", InMemoryRuntimeObserver())
    engine = AsyncLLM(v_config)

    logger.info(
        "FastInference API Server starting on http://%s:%s", args.host, args.port
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
