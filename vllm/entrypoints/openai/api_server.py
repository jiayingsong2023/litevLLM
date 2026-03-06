# SPDX-License-Identifier: Apache-2.0
import argparse
import asyncio
import json
import time
import os
import gguf
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
    parser.add_argument(
        "--policy-mode",
        type=str,
        choices=["auto", "aggressive", "stable"],
        default=os.getenv("FASTINF_POLICY_MODE", "auto"),
        help="Load-time execution policy selection mode.",
    )
    args = parser.parse_args()

    global engine
    from vllm.config import VllmConfig

    def _as_int_value(value):
        if isinstance(value, list):
            if not value:
                return 1
            return max(1, int(max(value)))
        return int(value)

    def build_hf_like_config_from_gguf(model_path: str):
        gguf_files = [f for f in os.listdir(model_path) if f.endswith(".gguf")]
        if not gguf_files:
            return None
        reader = gguf.GGUFReader(os.path.join(model_path, gguf_files[0]))
        architecture = str(reader.get_field("general.architecture").contents())

        def read_field_int(field_name: str, default: int) -> int:
            field = reader.get_field(field_name)
            if field is None:
                return default
            return _as_int_value(field.contents())

        def read_field_raw(field_name: str, default):
            field = reader.get_field(field_name)
            if field is None:
                return default
            return field.contents()

        class GGUFConfigFallback:
            pass

        cfg = GGUFConfigFallback()
        cfg.hidden_size = read_field_int(f"{architecture}.embedding_length", 4096)
        cfg.num_hidden_layers = read_field_int(f"{architecture}.block_count", 32)
        cfg.num_attention_heads = read_field_int(
            f"{architecture}.attention.head_count", 32
        )
        cfg.num_key_value_heads = read_field_int(
            f"{architecture}.attention.head_count_kv",
            cfg.num_attention_heads,
        )
        cfg.intermediate_size = read_field_int(f"{architecture}.feed_forward_length", 11008)
        cfg.max_position_embeddings = read_field_int(f"{architecture}.context_length", 32768)
        cfg.vocab_size = read_field_int(f"{architecture}.vocab_size", 32000)
        cfg.rms_norm_eps = 1e-6
        if architecture == "kimi-linear":
            cfg.architectures = ["KimiLinearForCausalLM"]
            kv_pattern = read_field_raw(f"{architecture}.attention.head_count_kv", [])
            if isinstance(kv_pattern, list):
                full_layers = [idx for idx, value in enumerate(kv_pattern) if int(value) > 0]
            else:
                full_layers = []
            cfg.linear_attn_config = {
                "full_attn_layers": full_layers,
                "kda_layers": [
                    idx for idx in range(cfg.num_hidden_layers) if idx not in set(full_layers)
                ],
            }
            cfg.first_k_dense_replace = read_field_int(
                f"{architecture}.leading_dense_block_count", 1
            )
            cfg.num_experts = read_field_int(f"{architecture}.expert_count", 0)
            cfg.num_experts_per_token = read_field_int(
                f"{architecture}.expert_used_count", 1
            )
            cfg.num_shared_experts = read_field_int(
                f"{architecture}.expert_shared_count", 0
            )
            cfg.moe_intermediate_size = read_field_int(
                f"{architecture}.expert_feed_forward_length", cfg.intermediate_size
            )
            cfg.qk_nope_head_dim = 128
            cfg.qk_rope_head_dim = 64
            cfg.v_head_dim = 128
            cfg.kv_lora_rank = read_field_int(f"{architecture}.attention.kv_lora_rank", 512)
        else:
            cfg.architectures = ["LlamaForCausalLM"]
        return cfg
    
    # Simple Mock Config for LitevLLM loading
    class FakeLiteConfig:
        def __init__(self, model_path, policy_mode: str):
            from transformers import AutoConfig
            import torch
            try:
                hf_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            except:
                fallback_cfg = build_hf_like_config_from_gguf(model_path)
                if fallback_cfg is not None:
                    hf_cfg = fallback_cfg
                else:
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
            self.runtime_policy_mode = policy_mode

    v_config = FakeLiteConfig(args.model, args.policy_mode)
    engine = AsyncLLM(v_config)
    
    logger.info(f"FastInference API Server starting on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
