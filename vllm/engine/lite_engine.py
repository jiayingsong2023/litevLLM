# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
import os
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
        
        # 2. Extract REAL dimensions from loaded model
        # Try to find actual dimensions from the first attention layer
        try:
            first_layer = None
            if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                first_layer = self.model.model.layers[0]
            elif hasattr(self.model, "layers"):
                first_layer = self.model.layers[0]
            
            if first_layer and hasattr(first_layer, "self_attn"):
                self.num_kv_heads = first_layer.self_attn.num_kv_heads
                self.head_size = first_layer.self_attn.head_dim
                print(f">>> LiteEngine: Detected Model Dimensions: {self.num_kv_heads} heads, {self.head_size} head_dim")
            else:
                self.num_kv_heads = self.model_config.get_num_kv_heads(None)
                self.head_size = self.model_config.get_head_size()
        except Exception as e:
            print(f">>> LiteEngine: Dimension detection failed ({e}), using defaults")
            self.num_kv_heads = self.model_config.get_num_kv_heads(None)
            self.head_size = self.model_config.get_head_size()

        self.num_layers = self.model_config.get_num_layers(None)
        self.max_model_len = min(self.model_config.get_max_model_len(), 4096)
        
        # 3. Pre-allocate Contiguous KV Cache
        self.max_active_requests = self.execution_policy.max_active_requests
        self.kv_dtype_str = os.environ.get("FASTINFERENCE_KV_FP8", "1")
        if self.kv_dtype_str == "1":
            print(f">>>> LiteEngine: KV Cache quantized to FP8 (e4m3fn) [DEFAULT]")
            self.kv_dtype = torch.float8_e4m3fn
        else:
            self.kv_dtype = torch.float16
            
        print(f">>>> LiteEngine: Allocating KV Cache ({self.max_active_requests} slots, {self.max_model_len} ctx, {self.num_layers} layers, dtype={self.kv_dtype})")
        
        self.kv_caches = []
        for _ in range(self.num_layers):
            # Shape: (MaxBatch, MaxLen, Heads, HeadDim)
            k = torch.zeros((self.max_active_requests, self.max_model_len, self.num_kv_heads, self.head_size), 
                          device=self.device, dtype=self.kv_dtype)
            v = torch.zeros((self.max_active_requests, self.max_model_len, self.num_kv_heads, self.head_size), 
                          device=self.device, dtype=self.kv_dtype)
            self.kv_caches.append((k, v))
        
        self._requests: Dict[str, Dict[str, Any]] = {}
        self._running_ids: List[str] = []
        self._free_slots = list(range(self.max_active_requests))
        self._request_slots: Dict[str, int] = {} # Map req_id -> slot_idx
        self._request_streams: Dict[str, asyncio.Queue] = {}

    def add_request(self, request_id: str, prompt: str, sampling_params: SamplingParams):
        if not self._free_slots:
            # Simple rejection if full. In real engine, we'd queue.
            print(f"!!! LiteEngine: Max capacity reached ({self.max_active_requests}), rejecting {request_id}")
            return

        effective_sampling_params = copy.deepcopy(sampling_params)
        max_tokens = effective_sampling_params.max_tokens or 16
        max_tokens = min(max_tokens, self.execution_policy.max_tokens_cap)

        input_ids = self.tokenizer.encode(prompt)
        slot_idx = self._free_slots.pop(0)
        
        self._requests[request_id] = {
            "input_ids": input_ids, 
            "generated_ids": [],
            "sampling_params": effective_sampling_params,
            "finished": False,
            "prompt": prompt,
            "slot_idx": slot_idx,
            "seq_len": 0, # Current length in KV cache
            "is_prefill": True
        }
        self._request_slots[request_id] = slot_idx
        self._running_ids.append(request_id)
        self._request_streams[request_id] = asyncio.Queue()

    async def get_request_stream(self, request_id: str) -> AsyncIterator[RequestOutput]:
        queue = self._request_streams[request_id]
        while True:
            output = await queue.get()
            yield output
            if output.finished: break

    def _free_request(self, rid: str):
        if rid in self._requests:
            slot = self._requests[rid]["slot_idx"]
            self._free_slots.append(slot)
            # Optional: Clear KV cache for this slot? Not strictly necessary if we track seq_len correctly.
            del self._requests[rid]
        if rid in self._request_slots: del self._request_slots[rid]
        if rid in self._running_ids: self._running_ids.remove(rid)

    @torch.inference_mode()
    def step(self) -> List[RequestOutput]:
        if not self._running_ids: return []
        
        # 1. Schedule: Separate Prefills and Decodes
        prefills = []
        decodes = []
        for rid in self._running_ids:
            if self._requests[rid]["is_prefill"]: prefills.append(rid)
            else: decodes.append(rid)
        
        results = []
        chunk_size = int(getattr(self.execution_policy, "chunked_prefill_size", 512))

        # 2. Execute
        # STRATEGY: One Prefill Chunk OR Batch All Decodes
        # This serializes Prefills but keeps Decodes at full BS=32 throughput.
        # Serialization is CRITICAL for AMD stability on long contexts.
        if prefills:
            # Run ONE chunk of ONE prefill
            rid = prefills[0]
            req = self._requests[rid]
            slot_idx = req["slot_idx"]
            all_input_ids = req["input_ids"]
            
            # Find how many tokens we have already processed
            processed_len = req["seq_len"]
            remaining_len = len(all_input_ids) - processed_len
            
            # Current chunk to process
            this_chunk_len = min(remaining_len, chunk_size)
            # If we are at the very last chunk, we generate 1 token, so seq_len increases by this_chunk_len
            # But the model sees 'this_chunk_len' tokens.
            
            curr_chunk_ids = all_input_ids[processed_len : processed_len + this_chunk_len]
            curr_input = torch.tensor([curr_chunk_ids], device=self.device)
            
            is_last_chunk = (processed_len + this_chunk_len >= len(all_input_ids))
            
            attn_metadata = {
                "slot_mapping": torch.tensor([slot_idx], device=self.device, dtype=torch.long),
                "seq_lens": torch.tensor([processed_len + this_chunk_len], device=self.device, dtype=torch.int32),
                "is_prefill": True,
                "kv_start_indices": torch.tensor([processed_len], device=self.device, dtype=torch.int32)
            }
            positions = torch.arange(processed_len, processed_len + this_chunk_len, device=self.device).unsqueeze(0)
            
            try:
                logits = self.model(curr_input, positions, self.kv_caches, attn_metadata)
                
                if is_last_chunk:
                    # Finally generate the first token
                    next_token = torch.argmax(logits[0, -1, :]).item()
                    req["generated_ids"].append(next_token)
                    req["seq_len"] = processed_len + this_chunk_len
                    req["is_prefill"] = False
                    self._process_completion(rid, next_token, results)
                else:
                    # Just update KV cache and stay in prefill mode
                    req["seq_len"] = processed_len + this_chunk_len
                    # No output for intermediate chunks in our simplified LiteEngine
            except Exception as e:
                print(f"!!! LiteEngine Error (Chunked Prefill): {e}"); import traceback; traceback.print_exc()
                self._free_request(rid)

        elif decodes:
            # Batch ALL decodes
            input_tokens = []
            slot_indices = []
            seq_lens = []
            pos_indices = []
            
            for rid in decodes:
                req = self._requests[rid]
                last_token = req["generated_ids"][-1]
                input_tokens.append([last_token])
                slot_indices.append(req["slot_idx"])
                
                current_len = req["seq_len"]
                seq_lens.append(current_len + 1) # Including new token
                pos_indices.append(current_len)
                
            curr_input = torch.tensor(input_tokens, device=self.device) # (B, 1)
            positions = torch.tensor(pos_indices, device=self.device).unsqueeze(1) # (B, 1)
            
            attn_metadata = {
                "slot_mapping": torch.tensor(slot_indices, device=self.device, dtype=torch.long),
                "seq_lens": torch.tensor(seq_lens, device=self.device, dtype=torch.int32),
                "is_prefill": False,
                "kv_start_indices": torch.tensor(pos_indices, device=self.device, dtype=torch.int32) # Write at pos
            }
            
            try:
                logits = self.model(curr_input, positions, self.kv_caches, attn_metadata)
                # logits: (B, 1, Vocab)
                next_tokens = torch.argmax(logits[:, -1, :], dim=-1).tolist()
                
                for i, rid in enumerate(decodes):
                    token = next_tokens[i]
                    self._requests[rid]["generated_ids"].append(token)
                    self._requests[rid]["seq_len"] += 1
                    self._process_completion(rid, token, results)
                    
            except Exception as e:
                print(f"!!! LiteEngine Error (Decode): {e}"); import traceback; traceback.print_exc()
                # Fail all involved? Or try to isolate? For Lite, just fail.
                for rid in decodes: self._free_request(rid)

        return results

    def _process_completion(self, rid, next_token, results):
        req = self._requests[rid]
        if next_token == getattr(self.tokenizer, "eos_token_id", -1) or len(req["generated_ids"]) >= (req["sampling_params"].max_tokens or 16):
            req["finished"] = True
        
        completion = CompletionOutput(index=0, text=self.tokenizer.decode(req["generated_ids"]), token_ids=req["generated_ids"], cumulative_logprob=0.0)
        out = RequestOutput(request_id=rid, prompt=req["prompt"], prompt_token_ids=req["input_ids"], outputs=[completion], finished=req["finished"])
        self._request_streams[rid].put_nowait(out)
        results.append(out)
        
        if req["finished"]:
            self._free_request(rid)
