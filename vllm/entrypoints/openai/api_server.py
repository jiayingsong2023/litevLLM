# SPDX-License-Identifier: Apache-2.0
import argparse
import asyncio
import json
import time
import os
from typing import AsyncGenerator, Dict, List, Optional, Union
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

from vllm.engine.async_llm import AsyncLLM
from vllm.sampling_params import SamplingParams
from vllm.logger import init_logger

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
    args = parser.parse_args()

    global engine
    from vllm.config import VllmConfig
    
    # Simple Mock Config for LitevLLM loading
    class FakeLiteConfig:
        def __init__(self, model_path):
            from transformers import AutoConfig
            import torch
            try:
                hf_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            except:
                class Simple: pass
                hf_cfg = Simple()
            
            self.model_config = type('obj', (object,), {
                'hf_config': hf_cfg, 'dtype': torch.float16,
                'max_model_len': 2048, 'model': model_path,
                'get_num_kv_heads': lambda x: getattr(hf_cfg, "num_key_value_heads", 1),
                'get_head_size': lambda: 128,
                'get_num_layers': lambda x: getattr(hf_cfg, "num_hidden_layers", 12),
            })
            if any(f.endswith(".gguf") for f in os.listdir(model_path)):
                from vllm.model_executor.layers.quantization.gguf import GGUFConfig
                self.quant_config = GGUFConfig()
            else: self.quant_config = None

    v_config = FakeLiteConfig(args.model)
    engine = AsyncLLM(v_config)
    
    logger.info(f"FastInference API Server starting on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
