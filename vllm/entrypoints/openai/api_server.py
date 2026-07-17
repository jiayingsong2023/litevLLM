# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import os
import time
import uuid
from collections.abc import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.engine.async_llm import AsyncLLM
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.serving.config_builder import build_vllm_config

logger = init_logger(__name__)
app = FastAPI()
engine: AsyncLLM | None = None


def _require_engine() -> AsyncLLM:
    if engine is None:
        raise HTTPException(status_code=503, detail="engine not initialized")
    return engine


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


def _parse_chat_message_content(
    messages: list[dict],
) -> tuple[str, dict | None]:
    if not messages:
        return "", None
    content = messages[-1].get("content")
    if isinstance(content, str):
        return content, None
    if not isinstance(content, list):
        raise HTTPException(
            status_code=400,
            detail="message content must be a string or content block list",
        )

    text_parts: list[str] = []
    images: list[dict[str, str]] = []
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
                    status_code=400, detail="image_url block must include non-empty url"
                )
            images.append({"image": url.strip()})
            text_parts.append("<image>")
            continue
        raise HTTPException(
            status_code=400, detail=f"unsupported message content type: {block_type}"
        )

    prompt = "\n".join(part for part in text_parts if part)
    multi_modal_data = {"image": images} if images else None
    return prompt, multi_modal_data


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


@app.get("/debug/runtime_stats")
async def get_runtime_stats():
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
    body = await request.json()
    model_name = body.get("model")
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens", 128)

    prompt, multi_modal_data = _parse_chat_message_content(messages)
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=body.get("temperature", 0.7),
        structured_outputs=_parse_structured_outputs(body),
    )
    request_id = _new_chat_request_id()

    async def generate_stream() -> AsyncGenerator[str, None]:
        last_text = ""
        async for output in runtime_engine.generate(
            prompt,
            sampling_params,
            request_id,
            multi_modal_data=multi_modal_data,
        ):
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
                        "finish_reason": "stop" if output.finished else None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    if stream:
        return StreamingResponse(generate_stream(), media_type="text/event-stream")
    else:
        final_output = None
        async for output in runtime_engine.generate(
            prompt,
            sampling_params,
            request_id,
            multi_modal_data=multi_modal_data,
        ):
            final_output = output

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
                        "finish_reason": "stop",
                    }
                ],
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="FastInference OpenAI API Server")
    parser.add_argument("--model", type=str, required=True, help="Path to the model.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--policy-mode",
        type=str,
        choices=["auto", "aggressive", "stable"],
        default=os.getenv("FASTINF_POLICY_MODE", "auto"),
        help="Load-time execution policy selection mode.",
    )
    args = parser.parse_args()

    global engine
    v_config = build_vllm_config(args.model, policy_mode=args.policy_mode)
    engine = AsyncLLM(v_config)

    logger.info(
        "FastInference API Server starting on http://%s:%s", args.host, args.port
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
