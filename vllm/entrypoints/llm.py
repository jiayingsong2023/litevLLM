# SPDX-License-Identifier: Apache-2.0
import torch
import gc
import os
import json
import time
from typing import List, Optional, Union
from vllm.sampling_params import SamplingParams
from vllm.outputs import RequestOutput, CompletionOutput
from vllm.model_executor.model_loader import get_model, get_tokenizer
from vllm.model_executor.layers.quantization.gguf import clear_gguf_cache
from transformers import AutoConfig

class LLM:
    def __init__(self, model: str, **kwargs):
        self.model_path = model; self.model = None; self.tokenizer = None; self._load_model()

    def _load_model(self):
        print(f">>> Loading model: {self.model_path}")
        try:
            hf_config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        except:
            config_path = os.path.join(self.model_path, "config.json")
            with open(config_path, "r") as f: data = json.load(f)
            class Simple: pass
            hf_config = Simple(); hf_config.__dict__.update(data.get("text_config", data))
            hf_config.architectures = data.get("architectures", [])
        
        n_heads = getattr(hf_config, "num_attention_heads", 32)
        n_kv_heads = getattr(hf_config, "num_key_value_heads", n_heads)
        h_size = getattr(hf_config, "hidden_size", 4096)
        n_layers = getattr(hf_config, "num_hidden_layers", 32)
        h_dim = h_size // n_heads
        
        class LiteVllmConfig:
            def __init__(self, path, hf_cfg):
                self.model_config = type('obj', (object,), {
                    'hf_config': hf_cfg, 'dtype': torch.float16,
                    'max_model_len': 4096, 'model': path,
                    'get_num_kv_heads': lambda x: n_kv_heads,
                    'get_head_size': lambda: h_dim,
                    'get_num_layers': lambda x: n_layers,
                })
                self.parallel_config = type('obj', (object,), {'tensor_parallel_size': 1, 'world_size': 1})
                if any(f.endswith(".gguf") for f in os.listdir(path)):
                    from vllm.model_executor.layers.quantization.gguf import GGUFConfig
                    self.quant_config = GGUFConfig()
                else: self.quant_config = None
        v_config = LiteVllmConfig(self.model_path, hf_config)
        self.model_cfg = v_config.model_config; self.model = get_model(v_config)
        self.tokenizer = get_tokenizer(v_config.model_config)

    @torch.inference_mode()
    def generate(self, prompts: List[str], sampling_params: Optional[SamplingParams] = None) -> List[RequestOutput]:
        batch_size = len(prompts)
        if sampling_params is None: sampling_params = SamplingParams()
        
        input_ids_list = [self.tokenizer.encode(p) for p in prompts]
        max_prompt_len = max(len(ids) for ids in input_ids_list)
        padded_ids = [ids + [self.tokenizer.pad_token_id or 0] * (max_prompt_len - len(ids)) for ids in input_ids_list]
        curr_input = torch.tensor(padded_ids, device="cuda")
        
        num_layers = self.model_cfg.get_num_layers(None); num_kv_heads = self.model_cfg.get_num_kv_heads(None); head_size = self.model_cfg.get_head_size()
        kv_caches = []
        for _ in range(num_layers):
            k = torch.zeros((128, 16, num_kv_heads, head_size), device="cuda", dtype=torch.float16)
            v = torch.zeros((128, 16, num_kv_heads, head_size), device="cuda", dtype=torch.float16)
            kv_caches.append((k, v))

        generated_ids = [[] for _ in range(batch_size)]; finished = [False] * batch_size
        for _ in range(sampling_params.max_tokens or 1):
            bsz, seq_len = curr_input.shape
            positions = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(bsz, -1)
            attn_metadata = {
                "slot_mapping": torch.arange(bsz, device="cuda", dtype=torch.int32),
                "seq_lens": torch.full((bsz,), seq_len, device="cuda", dtype=torch.int32)
            }
            logits = self.model(curr_input, positions, kv_caches, attn_metadata)
            next_tokens = torch.argmax(logits[:, -1, :], dim=-1)
            new_input_cols = []
            for i in range(batch_size):
                if not finished[i]:
                    token = next_tokens[i].item(); generated_ids[i].append(token)
                    if token == getattr(self.tokenizer, "eos_token_id", -1): finished[i] = True
                new_input_cols.append(next_tokens[i])
            if all(finished): break
            curr_input = torch.cat([curr_input, torch.stack(new_input_cols).unsqueeze(1)], dim=1)
        results = []
        for i in range(batch_size):
            completion = CompletionOutput(index=0, text=self.tokenizer.decode(generated_ids[i]), token_ids=generated_ids[i], cumulative_logprob=0.0)
            results.append(RequestOutput(request_id=str(time.time()), prompt=prompts[i], prompt_token_ids=input_ids_list[i], outputs=[completion], finished=True))
        return results
