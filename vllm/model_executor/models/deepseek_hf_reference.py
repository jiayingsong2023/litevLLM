# SPDX-License-Identifier: Apache-2.0
"""
Optional alignment with a local Hugging Face DeepSeek-V2-Lite-Chat checkout (bf16/fp16).

When `models/DeepSeek-V2-Lite-GGUF` and `models/DeepSeek-V2-Lite-Chat` sit as siblings under the
same parent directory, we can load tokenizer + rope/chat-related config from the HF tree so Lite
GGUF inference matches the reference checkpoint behavior more closely than a GGUF-only tree.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

# Env override: absolute path to the HF model directory (contains config.json + tokenizer).
_ENV_HF_CHAT = "FASTINFERENCE_DEEPSEEK_HF_CHAT"


def resolve_deepseek_hf_chat_dir(model_dir: str) -> Optional[str]:
    """
    If `.../models/<gguf-folder>` is next to `.../models/DeepSeek-V2-Lite-Chat`, return the Chat dir.
    """
    override = os.environ.get(_ENV_HF_CHAT, "").strip()
    if override and os.path.isdir(override) and os.path.isfile(os.path.join(override, "config.json")):
        return os.path.abspath(override)

    model_dir = os.path.abspath(model_dir)
    if not os.path.isdir(model_dir):
        return None
    # Only DeepSeek checkpoints may substitute the sibling HF Chat tree. Any other model
    # under the same parent (e.g. TinyLlama next to DeepSeek-V2-Lite-Chat) must not use it.
    base = os.path.basename(model_dir.rstrip(os.sep)).lower()
    if "deepseek" not in base:
        return None
    parent = os.path.dirname(model_dir)
    chat = os.path.join(parent, "DeepSeek-V2-Lite-Chat")
    cfg = os.path.join(chat, "config.json")
    tok_ok = os.path.isfile(os.path.join(chat, "tokenizer.json")) or os.path.isfile(
        os.path.join(chat, "tokenizer_config.json")
    )
    if os.path.isfile(cfg) and tok_ok:
        return chat
    return None


def merge_hf_reference_config_into_loaded_config(hf_config: Any, ref_json_path: str) -> None:
    """
    Copy RoPE / position / special-token fields from HF reference config.json onto hf_config.
    Does not overwrite architecture sizes that must match GGUF tensors (hidden_size, num_layers, ...).
    """
    with open(ref_json_path, "r", encoding="utf-8") as f:
        ref = json.load(f)
    if ref.get("model_type") != "deepseek_v2":
        return

    # Merge rope dicts onto GGUF-loaded config so we keep fields the GGUF tree has
    # (e.g. rope_type) while overlaying HF reference numeric/token fields.
    if isinstance(ref.get("rope_parameters"), dict):
        cur = getattr(hf_config, "rope_parameters", None)
        merged = dict(cur) if isinstance(cur, dict) else {}
        merged.update(ref["rope_parameters"])
        setattr(hf_config, "rope_parameters", merged)
    if isinstance(ref.get("rope_scaling"), dict):
        cur = getattr(hf_config, "rope_scaling", None)
        merged = dict(cur) if isinstance(cur, dict) else {}
        merged.update(ref["rope_scaling"])
        setattr(hf_config, "rope_scaling", merged)

    for key in (
        "rope_theta",
        "bos_token_id",
        "eos_token_id",
        "max_position_embeddings",
        "tie_word_embeddings",
    ):
        if key in ref:
            setattr(hf_config, key, ref[key])
