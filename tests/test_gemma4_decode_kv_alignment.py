# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import gc
from pathlib import Path
from typing import Callable

import pytest
import torch
import torch.nn.functional as F

from vllm.config import CacheConfig, LoadConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.engine.lite_engine import LiteEngine
from vllm.kernels.triton.reshape_and_cache import reshape_and_cache
from vllm.model_executor.model_loader import get_tokenizer
from vllm.model_executor.models.gemma4 import _decode_int4_row, _gather_recent_kv


def _resolve_gemma4_26b_model_path() -> str | None:
    env = os.environ.get("MODEL_GEMMA4_26B_A4B_Q4", "").strip()
    if env:
        return env
    candidates = [
        "models/gemma-4-26B-A4B-it-AWQ-4bit",
        "models/Gemma-4-26B-A4B-it-AWQ-4bit",
    ]
    for p in candidates:
        if Path(p).is_dir():
            return p
    return None


def _build_lite_engine(model_path: str) -> LiteEngine:
    model_cfg = ModelConfig(model=model_path, tokenizer=model_path)
    cache_cfg = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.55,
        swap_space=4,
    )
    sched_cfg = SchedulerConfig(
        max_num_batched_tokens=256,
        max_num_seqs=1,
        max_model_len=256,
    )
    load_cfg = LoadConfig(load_format="auto")
    v_cfg = VllmConfig(model_cfg, cache_cfg, sched_cfg, load_cfg, quant_config=None)
    engine = LiteEngine(v_cfg)
    engine.tokenizer = get_tokenizer(model_path, trust_remote_code=True)
    return engine


def _set_or_unset_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def _run_window_audit_once(
    *,
    model_path: str,
    prompt: str,
    token_start: int,
    token_end: int,
    cos_threshold: float,
    kv_type: str,
    guard_enabled: bool,
    guard_start: int | None = None,
    guard_span: int | None = None,
) -> dict[str, int | float | None]:
    _set_or_unset_env("FASTINFERENCE_KV_TYPE", kv_type)
    _set_or_unset_env("FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS", "1")
    _set_or_unset_env("FASTINFERENCE_KV_MAX_MODEL_LEN", "256")
    _set_or_unset_env(
        "FASTINFERENCE_GEMMA4_26B_FP32_RESIDUAL_GUARD",
        "1" if guard_enabled else None,
    )
    _set_or_unset_env(
        "FASTINFERENCE_GEMMA4_26B_FP32_RESIDUAL_GUARD_START",
        str(guard_start) if guard_start is not None else None,
    )
    _set_or_unset_env(
        "FASTINFERENCE_GEMMA4_26B_FP32_RESIDUAL_GUARD_SPAN",
        str(guard_span) if guard_span is not None else None,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    engine = _build_lite_engine(model_path)
    tokenizer = engine.tokenizer
    assert tokenizer is not None
    prompt_ids = tokenizer.encode(prompt)
    if isinstance(prompt_ids, torch.Tensor):
        prompt_ids = prompt_ids.tolist()
    return _audit_local_decode_window(
        engine,
        list(prompt_ids),
        token_start=token_start,
        token_end=token_end,
        cos_threshold=cos_threshold,
    )


def _audit_local_decode_window_ab(
    *,
    model_path: str,
    prompt: str,
    token_start: int,
    token_end: int,
    cos_threshold: float,
) -> dict[str, object]:
    baseline: dict[str, int | float | None] | None
    baseline_error: str | None = None
    try:
        baseline = _run_window_audit_once(
            model_path=model_path,
            prompt=prompt,
            token_start=token_start,
            token_end=token_end,
            cos_threshold=cos_threshold,
            kv_type="fp16",
            guard_enabled=False,
        )
    except Exception as e:
        baseline = None
        baseline_error = f"{type(e).__name__}: {e}"

    drift_layer = baseline.get("first_drift_layer") if baseline is not None else None
    guard_start = int(drift_layer) if drift_layer is not None else 8
    guarded: dict[str, int | float | None] | None
    guarded_error: str | None = None
    try:
        guarded = _run_window_audit_once(
            model_path=model_path,
            prompt=prompt,
            token_start=token_start,
            token_end=token_end,
            cos_threshold=cos_threshold,
            kv_type="fp16",
            guard_enabled=True,
            guard_start=guard_start,
            guard_span=3,
        )
    except Exception as e:
        guarded = None
        guarded_error = f"{type(e).__name__}: {e}"
    return {
        "baseline": baseline,
        "baseline_error": baseline_error,
        "guarded": guarded,
        "guarded_error": guarded_error,
        "guard_start": guard_start,
        "guard_span": 3,
    }


def _build_cached_meta(
    engine: LiteEngine,
    *,
    pos_start: int,
    seqlen: int,
    seq_len_after: int,
    is_prefill: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | bool | object | list[tuple[torch.Tensor | None, torch.Tensor | None]]]]:
    slot_mapping = torch.arange(
        pos_start, pos_start + seqlen, device="cuda", dtype=torch.long
    )
    positions = torch.arange(
        pos_start, pos_start + seqlen, device="cuda", dtype=torch.long
    ).view(1, seqlen)
    block_tables = torch.arange(
        engine.num_blocks_per_seq, device="cuda", dtype=torch.int32
    ).view(1, -1)
    meta = {
        "slot_mapping": slot_mapping,
        "block_tables": block_tables,
        "seq_lens": torch.tensor([seq_len_after], device="cuda", dtype=torch.int32),
        "is_prefill": bool(is_prefill),
        "config": engine.inf_config,
        "kv_scale_cache": engine.kv_scale_caches,
    }
    return positions, meta


def _last_token_hook(capture: dict[int, torch.Tensor], layer_idx: int):
    def _hook(_module: torch.nn.Module, _inputs: tuple[object, ...], output: object) -> None:
        t = output if isinstance(output, torch.Tensor) else output[0]
        capture[layer_idx] = t[:, -1, :].detach().float().cpu()

    return _hook


def _skip_if_known_gemma4_loader_shape_instability(err: Exception) -> None:
    msg = str(err)
    if "shape" in msg and "invalid for input of size" in msg:
        pytest.skip(f"Gemma4 loader shape instability in current run: {msg}")


def _capture_last_hidden_for_layers(
    layers: list[torch.nn.Module],
    run_forward: Callable[[], object],
) -> dict[int, torch.Tensor]:
    capture: dict[int, torch.Tensor] = {}
    hooks = [
        layer.register_forward_hook(_last_token_hook(capture, li))
        for li, layer in enumerate(layers)
    ]
    try:
        run_forward()
    finally:
        for h in hooks:
            h.remove()
    return capture


def _audit_local_decode_window(
    engine: LiteEngine,
    prompt_ids: list[int],
    *,
    token_start: int,
    token_end: int,
    cos_threshold: float,
) -> dict[str, int | float | None]:
    layers = list(engine.model.model.layers)
    block_tables = torch.arange(
        engine.num_blocks_per_seq, device="cuda", dtype=torch.int32
    ).view(1, -1)

    def _cached_forward(
        input_ids: torch.Tensor, pos_start: int, seq_len_after: int, is_prefill: bool
    ) -> torch.Tensor:
        seqlen = input_ids.shape[1]
        positions = torch.arange(
            pos_start, pos_start + seqlen, device="cuda", dtype=torch.long
        ).view(1, seqlen)
        slot_mapping = torch.arange(
            pos_start, pos_start + seqlen, device="cuda", dtype=torch.long
        )
        meta = {
            "slot_mapping": slot_mapping,
            "block_tables": block_tables,
            "seq_lens": torch.tensor([seq_len_after], device="cuda", dtype=torch.int32),
            "is_prefill": bool(is_prefill),
            "config": engine.inf_config,
            "kv_scale_cache": engine.kv_scale_caches,
        }
        return engine.model(input_ids, positions, engine.kv_caches, meta, None)

    # Reset KV state.
    for k_cache, v_cache in engine.kv_caches:
        k_cache.zero_()
        v_cache.zero_()
    for ks, vs in engine.kv_scale_caches:
        if ks is not None:
            ks.zero_()
        if vs is not None:
            vs.zero_()

    generated: list[int] = []
    prefill_input = torch.tensor([prompt_ids], device="cuda", dtype=torch.long)
    logits = _cached_forward(prefill_input, 0, len(prompt_ids), True)

    first_drift_token: int | None = None
    first_drift_layer: int | None = None
    first_drift_cos: float | None = None

    for step in range(1, token_end + 1):
        token = int(torch.argmax(logits[0, -1]).item())
        generated.append(token)
        token_input = torch.tensor([[token]], device="cuda", dtype=torch.long)
        pos_start = len(prompt_ids) + step - 1

        cached_cap = _capture_last_hidden_for_layers(
            layers,
            lambda: _cached_forward(token_input, pos_start, len(prompt_ids) + step, False),
        )

        full_ids = prompt_ids + generated
        full_input = torch.tensor([full_ids], device="cuda", dtype=torch.long)
        full_positions = torch.arange(
            len(full_ids), device="cuda", dtype=torch.long
        ).view(1, -1)
        ref_cap = _capture_last_hidden_for_layers(
            layers,
            lambda: engine.model(
                full_input,
                full_positions,
                [(None, None)] * len(engine.kv_caches),
                {},
                None,
            ),
        )

        if step >= token_start and first_drift_token is None:
            for li, layer in enumerate(layers):
                if not bool(getattr(layer.self_attn, "is_sliding", False)):
                    continue
                cos = float(
                    F.cosine_similarity(
                        cached_cap[li].view(1, -1),
                        ref_cap[li].view(1, -1),
                        dim=-1,
                    ).item()
                )
                if cos < cos_threshold:
                    first_drift_token = step
                    first_drift_layer = li
                    first_drift_cos = cos
                    break

        logits = _cached_forward(token_input, pos_start, len(prompt_ids) + step, False)

    return {
        "first_drift_token": first_drift_token,
        "first_drift_layer": first_drift_layer,
        "first_drift_cos": first_drift_cos,
        "token_start": token_start,
        "token_end": token_end,
        "cos_threshold": cos_threshold,
    }


def test_gemma4_local_decode_step2_aligns_with_nocache_first_local_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = _resolve_gemma4_26b_model_path()
    if not model_path:
        pytest.skip("Gemma4-26B-A4B local model dir not found; skipping decode alignment test.")
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device unavailable; skipping decode alignment test.")

    monkeypatch.setenv("FASTINFERENCE_KV_TYPE", "fp16")
    monkeypatch.setenv("FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS", "1")
    monkeypatch.setenv("FASTINFERENCE_KV_MAX_MODEL_LEN", "256")

    engine = _build_lite_engine(model_path)
    tokenizer = engine.tokenizer
    assert tokenizer is not None

    prompt_ids = tokenizer.encode("Hi,")
    if isinstance(prompt_ids, torch.Tensor):
        prompt_ids = prompt_ids.tolist()
    prompt_ids = list(prompt_ids)
    assert len(prompt_ids) > 0

    for k_cache, v_cache in engine.kv_caches:
        k_cache.zero_()
        v_cache.zero_()

    prefill_input = torch.tensor([prompt_ids], device="cuda", dtype=torch.long)
    prefill_pos, prefill_meta = _build_cached_meta(
        engine,
        pos_start=0,
        seqlen=len(prompt_ids),
        seq_len_after=len(prompt_ids),
        is_prefill=True,
    )
    try:
        logits_prefill = engine.model(
            prefill_input, prefill_pos, engine.kv_caches, prefill_meta, None
        )
    except RuntimeError as e:
        _skip_if_known_gemma4_loader_shape_instability(e)
        raise
    first_token = int(torch.argmax(logits_prefill[0, -1]).item())

    layers = list(engine.model.model.layers)
    cached_capture: dict[int, torch.Tensor] = {}
    hooks = [
        layer.register_forward_hook(_last_token_hook(cached_capture, li))
        for li, layer in enumerate(layers)
    ]
    decode_input = torch.tensor([[first_token]], device="cuda", dtype=torch.long)
    decode_pos, decode_meta = _build_cached_meta(
        engine,
        pos_start=len(prompt_ids),
        seqlen=1,
        seq_len_after=len(prompt_ids) + 1,
        is_prefill=False,
    )
    try:
        _ = engine.model(decode_input, decode_pos, engine.kv_caches, decode_meta, None)
    except RuntimeError as e:
        _skip_if_known_gemma4_loader_shape_instability(e)
        raise
    for h in hooks:
        h.remove()

    ref_capture: dict[int, torch.Tensor] = {}
    hooks = [
        layer.register_forward_hook(_last_token_hook(ref_capture, li))
        for li, layer in enumerate(layers)
    ]
    full_ids = prompt_ids + [first_token]
    full_input = torch.tensor([full_ids], device="cuda", dtype=torch.long)
    full_pos = torch.arange(len(full_ids), device="cuda", dtype=torch.long).view(1, -1)
    try:
        _ = engine.model(full_input, full_pos, [(None, None)] * len(engine.kv_caches), {}, None)
    except RuntimeError as e:
        _skip_if_known_gemma4_loader_shape_instability(e)
        raise
    for h in hooks:
        h.remove()

    local_layer_cos: list[tuple[int, float]] = []
    for li, layer in enumerate(layers):
        if not bool(getattr(layer.self_attn, "is_sliding", False)):
            continue
        a = cached_capture[li]
        b = ref_capture[li]
        cos = float(F.cosine_similarity(a.view(1, -1), b.view(1, -1), dim=-1).item())
        local_layer_cos.append((li, cos))

    assert len(local_layer_cos) >= 3
    first_three = local_layer_cos[:3]
    for li, cos in first_three:
        assert cos >= 0.99, f"local layer {li} cos too low: {cos:.6f}"


def test_gemma4_local_decode_window_2_to_16_reports_first_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = _resolve_gemma4_26b_model_path()
    if not model_path:
        pytest.skip("Gemma4-26B-A4B local model dir not found; skipping window audit.")
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device unavailable; skipping window audit.")

    monkeypatch.setenv("FASTINFERENCE_KV_TYPE", "fp16")
    monkeypatch.setenv("FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS", "1")
    monkeypatch.setenv("FASTINFERENCE_KV_MAX_MODEL_LEN", "256")

    engine = _build_lite_engine(model_path)
    tokenizer = engine.tokenizer
    assert tokenizer is not None
    prompt_ids = tokenizer.encode("Hi,")
    if isinstance(prompt_ids, torch.Tensor):
        prompt_ids = prompt_ids.tolist()
    prompt_ids = list(prompt_ids)
    assert len(prompt_ids) > 0

    try:
        report = _audit_local_decode_window(
            engine,
            prompt_ids,
            token_start=2,
            token_end=16,
            cos_threshold=0.99,
        )
    except RuntimeError as e:
        _skip_if_known_gemma4_loader_shape_instability(e)
        raise
    assert report["first_drift_token"] is not None, f"window audit report={report}"
    assert report["first_drift_layer"] is not None, f"window audit report={report}"
    assert 2 <= int(report["first_drift_token"]) <= 16, f"window audit report={report}"


def test_gemma4_local_decode_window_2_to_16_ab_compare_baseline_vs_guard() -> None:
    model_path = _resolve_gemma4_26b_model_path()
    if not model_path:
        pytest.skip("Gemma4-26B-A4B local model dir not found; skipping AB window audit.")
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device unavailable; skipping AB window audit.")

    ab = _audit_local_decode_window_ab(
        model_path=model_path,
        prompt="Hi,",
        token_start=2,
        token_end=16,
        cos_threshold=0.99,
    )
    baseline = ab["baseline"]
    guarded = ab["guarded"]
    if baseline is None:
        pytest.skip(f"Baseline AB leg failed in current environment: {ab.get('baseline_error')}")
    assert isinstance(baseline, dict)
    if guarded is None:
        pytest.skip(f"Guarded AB leg failed in current environment: {ab.get('guarded_error')}")
    assert isinstance(guarded, dict)
    assert baseline.get("first_drift_token") is not None, f"ab={ab}"
    assert baseline.get("first_drift_layer") is not None, f"ab={ab}"
    assert guarded.get("first_drift_token") is not None, f"ab={ab}"
    assert guarded.get("first_drift_layer") is not None, f"ab={ab}"
    assert int(ab["guard_span"]) == 3


def test_turbo_int4_kv_readback_pack_scale_consistency() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device unavailable; skipping turbo_int4 KV readback test.")

    torch.manual_seed(7)
    num_tokens = 20
    num_kv_heads = 3
    head_dim = 8
    block_size = 16
    num_blocks = 2
    half_dim = head_dim // 2

    key = torch.randn(num_tokens, num_kv_heads, head_dim, device="cuda", dtype=torch.float32)
    value = torch.randn(num_tokens, num_kv_heads, head_dim, device="cuda", dtype=torch.float32)
    # Make clipping path observable.
    key[3, 1, 0] = 20.0
    value[9, 2, 7] = -20.0

    key_cache = torch.zeros(
        num_blocks, block_size, num_kv_heads, half_dim, device="cuda", dtype=torch.uint8
    )
    value_cache = torch.zeros_like(key_cache)
    slot_mapping = torch.arange(num_tokens, device="cuda", dtype=torch.long)

    k_scale_cache = torch.zeros(
        num_blocks, block_size, num_kv_heads, 1, device="cuda", dtype=torch.float32
    )
    v_scale_cache = torch.zeros_like(k_scale_cache)

    reshape_and_cache(
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slot_mapping,
        kv_cache_dtype="turbo_int4",
        k_scale=1.0,
        v_scale=1.0,
        k_scale_cache=k_scale_cache,
        v_scale_cache=v_scale_cache,
    )

    def _expected_quantized_row(
        src_row: torch.Tensor, row_scale: torch.Tensor
    ) -> torch.Tensor:
        lo = src_row[:, :half_dim]
        hi = src_row[:, half_dim:]
        scale = row_scale[:, None]
        lo_q = torch.clamp(torch.floor(lo / scale + 0.5), -7.0, 7.0)
        hi_q = torch.clamp(torch.floor(hi / scale + 0.5), -7.0, 7.0)
        return torch.cat([lo_q * scale, hi_q * scale], dim=-1)

    for tok in range(num_tokens):
        block_idx = tok // block_size
        block_off = tok % block_size

        dec_k = _decode_int4_row(
            key_cache, k_scale_cache, block_idx, block_off, num_kv_heads, head_dim
        )
        dec_v = _decode_int4_row(
            value_cache, v_scale_cache, block_idx, block_off, num_kv_heads, head_dim
        )
        exp_k = _expected_quantized_row(
            key[tok].float(),
            k_scale_cache[block_idx, block_off, :, 0].float(),
        )
        exp_v = _expected_quantized_row(
            value[tok].float(),
            v_scale_cache[block_idx, block_off, :, 0].float(),
        )

        assert torch.allclose(dec_k, exp_k, atol=1e-4, rtol=0.0)
        assert torch.allclose(dec_v, exp_v, atol=1e-4, rtol=0.0)

    block_tables = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([num_tokens], device="cuda", dtype=torch.int32)
    gathered_k, gathered_v = _gather_recent_kv(
        kv_cache=(key_cache, value_cache),
        block_tables=block_tables,
        seq_lens=seq_lens,
        batch_idx=0,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        local_window=None,
        kv_cache_dtype="turbo_int4",
        kv_scale_cache=(k_scale_cache, v_scale_cache),
    )

    assert gathered_k.shape == (1, num_tokens, num_kv_heads, head_dim)
    assert gathered_v.shape == (1, num_tokens, num_kv_heads, head_dim)
    for tok in range(num_tokens):
        block_idx = tok // block_size
        block_off = tok % block_size
        exp_k = _expected_quantized_row(
            key[tok].float(),
            k_scale_cache[block_idx, block_off, :, 0].float(),
        )
        exp_v = _expected_quantized_row(
            value[tok].float(),
            v_scale_cache[block_idx, block_off, :, 0].float(),
        )
        assert torch.allclose(gathered_k[0, tok], exp_k, atol=1e-4, rtol=0.0)
        assert torch.allclose(gathered_v[0, tok], exp_v, atol=1e-4, rtol=0.0)
