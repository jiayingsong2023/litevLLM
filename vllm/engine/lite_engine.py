# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
import torch
import torch.nn as nn
import copy
from typing import Any, AsyncIterator, Dict, List, Optional, Union, Tuple
from vllm.config import VllmConfig
from vllm.engine.loadtime_policy import select_loadtime_policy
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.outputs import RequestOutput, CompletionOutput
from vllm.sampling_params import SamplingParams

logger = init_logger(__name__)

class LiteEngine:
    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = torch.device("cuda:0")
        requested_policy_mode = str(
            getattr(vllm_config, "runtime_policy_mode", "auto")
        ).lower()
        
        # 1. Load Model
        print(f">>> LiteEngine: Loading {self.model_config.model}...")
        self.model = get_model(vllm_config=self.vllm_config)
        self.tokenizer = None 
        self.execution_policy = select_loadtime_policy(
            model_config=self.model_config,
            quant_config=getattr(vllm_config, "quant_config", None),
            policy_mode=requested_policy_mode,  # type: ignore[arg-type]
        )
        
        # 2. Extract REAL dimensions from model_config
        self.num_layers = self.model_config.get_num_layers(None)
        self.num_kv_heads = self.model_config.get_num_kv_heads(None)
        self.head_size = self.model_config.get_head_size()
        
        print(f">>> LiteEngine: Allocating KV Cache ({self.num_layers} layers, {self.num_kv_heads} heads, {self.head_size} head_dim)")
        self.kv_caches = []
        for _ in range(self.num_layers):
            k = torch.zeros((128, 16, self.num_kv_heads, self.head_size), device=self.device, dtype=torch.float16)
            v = torch.zeros((128, 16, self.num_kv_heads, self.head_size), device=self.device, dtype=torch.float16)
            self.kv_caches.append((k, v))
        
        self._requests: Dict[str, Dict[str, Any]] = {}
        self._running_ids: List[str] = []
        self._request_streams: Dict[str, asyncio.Queue] = {}

    def add_request(self, request_id: str, prompt: str, sampling_params: SamplingParams):
        effective_sampling_params = copy.deepcopy(sampling_params)
        max_tokens = effective_sampling_params.max_tokens
        if max_tokens is None:
            effective_sampling_params.max_tokens = self.execution_policy.max_tokens_cap
        else:
            effective_sampling_params.max_tokens = min(
                max_tokens,
                self.execution_policy.max_tokens_cap,
            )

        input_ids = self.tokenizer.encode(prompt)
        self._requests[request_id] = {
            "input_ids": input_ids, "generated_ids": [],
            "sampling_params": effective_sampling_params,
            "finished": False,
            "prompt": prompt,
        }
        self._running_ids.append(request_id)
        self._request_streams[request_id] = asyncio.Queue()

    async def get_request_stream(self, request_id: str) -> AsyncIterator[RequestOutput]:
        queue = self._request_streams[request_id]
        while True:
            output = await queue.get()
            yield output
            if output.finished: break

    @torch.inference_mode()
    def step(self) -> List[RequestOutput]:
        if not self._running_ids: return []
        results = []
        active_ids = list(self._running_ids[: self.execution_policy.max_active_requests])
        for rid in active_ids:
            req = self._requests[rid]
            try:
                tokens = req["input_ids"] + req["generated_ids"]
                curr_input = torch.tensor([tokens], device=self.device)
                seq_len = curr_input.shape[1]
                positions = torch.arange(seq_len, device=self.device).unsqueeze(0)
                attn_metadata = {"slot_mapping": torch.arange(seq_len, device=self.device, dtype=torch.int32), "seq_lens": torch.tensor([seq_len], device=self.device, dtype=torch.int32)}
                logits = self.model(curr_input, positions, self.kv_caches, attn_metadata)
                next_token = torch.argmax(logits[0, -1, :]).item()
                req["generated_ids"].append(next_token)
                if next_token == getattr(self.tokenizer, "eos_token_id", -1) or len(req["generated_ids"]) >= (req["sampling_params"].max_tokens or 16):
                    req["finished"] = True; self._running_ids.remove(rid)
                completion = CompletionOutput(index=0, text=self.tokenizer.decode(req["generated_ids"]), token_ids=req["generated_ids"], cumulative_logprob=0.0)
                out = RequestOutput(request_id=rid, prompt=req["prompt"], prompt_token_ids=req["input_ids"], outputs=[completion], finished=req["finished"])
                self._request_streams[rid].put_nowait(out); results.append(out)
            except Exception as e:
                print(f"!!! LiteEngine Error: {e}"); import traceback; traceback.print_exc()
                req["finished"] = True; self._running_ids.remove(rid)
        return results
