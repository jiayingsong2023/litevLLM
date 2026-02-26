# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import pickle
from _hashlib import HASH, UnsupportedDigestmodError
from collections.abc import Callable
from typing import Any

import cbor2

try:
    # It is important that this remains an optional dependency.
    # It would not be allowed in environments with strict security controls,
    # so it's best not to have it installed when not in use.
    import xxhash as _xxhash

    if not hasattr(_xxhash, "xxh3_128_digest"):
        _xxhash = None
except ImportError:  # pragma: no cover
    _xxhash = None

def sha256(input: Any) -> bytes:
    input_bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(input_bytes).digest()

def sha256_cbor(input: Any) -> bytes:
    input_bytes = cbor2.dumps(input, canonical=True)
    return hashlib.sha256(input_bytes).digest()

def _xxhash_digest(input_bytes: bytes) -> bytes:
    if _xxhash is None:
        raise ModuleNotFoundError(
            "xxhash is required for the 'xxhash' prefix caching hash algorithms. "
            "Install it via `pip install xxhash`."
        )
    return _xxhash.xxh3_128_digest(input_bytes)

def xxhash(input: Any) -> bytes:
    input_bytes = cbor2.dumps(input, canonical=True)
    return _xxhash_digest(input_bytes)

def get_hash_fn_by_name(hash_fn_name: str) -> Callable[[Any], bytes]:
    if hash_fn_name == "sha256":
        return sha256
    if hash_fn_name == "sha256_cbor":
        return sha256_cbor
    if hash_fn_name == "xxhash":
        return xxhash
    if hash_fn_name == "xxhash_cbor":
        return xxhash_cbor

    raise ValueError(f"Unsupported hash function: {hash_fn_name}")

def safe_hash(data: bytes, usedforsecurity: bool = True) -> HASH:
    try:
        return hashlib.md5(data, usedforsecurity=usedforsecurity)
    except (UnsupportedDigestmodError, ValueError):
        return hashlib.sha256(data)
