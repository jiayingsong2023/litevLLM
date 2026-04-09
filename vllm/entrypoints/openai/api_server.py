# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import time
import os
from typing import AsyncGenerator, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

from vllm.engine.async_llm import AsyncLLM
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.logger import init_logger
from vllm.serving.config_builder import build_vllm_config

logger = init_logger(__name__)
app = FastAPI()
engine: Optional[AsyncLLM] = None


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
            choice=list(raw_structured["choice"]) if raw_structured.get("choice") is not None else None,
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
            raise HTTPException(status_code=400, detail="response_format.json_schema must be an object")
        schema = json_schema.get("schema") or json_schema.get("json_schema")
        if not isinstance(schema, dict):
            raise HTTPException(status_code=400, detail="response_format.json_schema.schema must be an object")
        return StructuredOutputsParams(json=schema)
    if rf_type == "text" or rf_type is None:
        return None
    raise HTTPException(status_code=400, detail=f"unsupported response_format.type: {rf_type}")

@app.get("/v1/models")
async def show_available_models():
    model_config = await _require_engine().get_model_config()
    return JSONResponse(content={
        "object": "list",
        "data": [{"id": model_config.model, "object": "model", "created": int(time.time()), "owned_by": "fastinference"}]
    })


@app.get("/debug/runtime_stats")
async def get_runtime_stats():
    runtime_engine = _require_engine()
    model_config = await runtime_engine.get_model_config()
    return JSONResponse(
        content={
            "model": getattr(model_config, "model", None),
            "stats": runtime_engine.stats(),
        }
    )


@app.post("/debug/runtime_stats/reset")
async def reset_runtime_stats(clear_prefix_cache: bool = False):
    runtime_engine = _require_engine()
    runtime_engine.reset_stats(clear_prefix_cache=clear_prefix_cache)
    model_config = await runtime_engine.get_model_config()
    return JSONResponse(
        content={
            "model": getattr(model_config, "model", None),
            "stats": runtime_engine.stats(),
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
    
    prompt = messages[-1]["content"] if messages else ""
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=body.get("temperature", 0.7),
        structured_outputs=_parse_structured_outputs(body),
    )
    request_id = f"chat-{int(time.time())}"
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        last_text = ""
        async for output in runtime_engine.generate(prompt, sampling_params, request_id):
            current_text = output.outputs[0].text if output.outputs else ""
            delta_text = current_text[len(last_text):] if current_text.startswith(last_text) else current_text
            last_text = current_text
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "delta": {"content": delta_text}, "finish_reason": "stop" if output.finished else None}]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    if stream:
        return StreamingResponse(generate_stream(), media_type="text/event-stream")
    else:
        final_output = None
        async for output in runtime_engine.generate(prompt, sampling_params, request_id):
            final_output = output
        
        return JSONResponse(content={
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": final_output.outputs[0].text}, "finish_reason": "stop"}]
        })

def main():
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
    
    logger.info(f"FastInference API Server starting on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
