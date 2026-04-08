# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import time
import os
from typing import AsyncGenerator, Optional
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

from vllm.engine.async_llm import AsyncLLM
from vllm.sampling_params import SamplingParams
from vllm.logger import init_logger
from vllm.serving.config_builder import build_vllm_config

logger = init_logger(__name__)
app = FastAPI()
engine: Optional[AsyncLLM] = None

@app.get("/v1/models")
async def show_available_models():
    model_config = await engine.get_model_config()
    return JSONResponse(content={
        "object": "list",
        "data": [{"id": model_config.model, "object": "model", "created": int(time.time()), "owned_by": "fastinference"}]
    })

@app.post("/v1/chat/completions")
async def create_chat_completion(request: Request):
    body = await request.json()
    model_name = body.get("model")
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens", 128)
    
    prompt = messages[-1]["content"] if messages else ""
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=body.get("temperature", 0.7))
    request_id = f"chat-{int(time.time())}"
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        async for output in engine.generate(prompt, sampling_params, request_id):
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "delta": {"content": output.outputs[0].text if output.outputs else ""}, "finish_reason": "stop" if output.finished else None}]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    if stream:
        return StreamingResponse(generate_stream(), media_type="text/event-stream")
    else:
        final_output = None
        async for output in engine.generate(prompt, sampling_params, request_id):
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
