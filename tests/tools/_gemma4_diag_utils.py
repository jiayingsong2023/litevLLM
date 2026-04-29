# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for Gemma4 diagnostic warn-only tests."""
from __future__ import annotations

import re
import warnings
from typing import Any


def parse_strict_metrics(text: str) -> dict[str, Any]:
    cos_re = re.search(r"Prefill Logits -> CosSim:\s*([0-9.]+), MaxErr:\s*([0-9.]+)", text)
    tok_re = re.search(
        r"Prefill Token:\s*HF\(argmax\)=(-?\d+)\s*\|\s*Lite\(engine\)=(-?\d+)\s*\|\s*Lite\(argmax logits\)=(-?\d+)",
        text,
    )
    if cos_re is None or tok_re is None:
        raise ValueError("Could not parse strict audit metrics from output.")
    return {
        "cos_sim": float(cos_re.group(1)),
        "max_err": float(cos_re.group(2)),
        "hf_argmax": int(tok_re.group(1)),
        "lite_engine_argmax": int(tok_re.group(2)),
        "lite_logits_argmax": int(tok_re.group(3)),
    }


def parse_drift_metrics(text: str) -> dict[str, Any]:
    token_map: dict[int, int] = {}
    tm_re = re.search(r"\[Drift\] token_to_step=\{([^}]*)\}", text)
    if tm_re is not None:
        for pair in tm_re.group(1).split(","):
            pair = pair.strip()
            if not pair:
                continue
            k, v = pair.split(":")
            token_map[int(k.strip())] = int(v.strip())

    metrics: dict[int, dict[str, dict[str, float]]] = {}
    cur_token: int | None = None
    token_re = re.compile(r"^\[token=(\d+)\]")
    line_re = re.compile(
        r"^\s+(local|full):\s+cos_to_t1 mean=([0-9.]+)\s+min=([0-9.]+)\s+"
        r"cos_to_prev mean=([0-9.]+)\s+min=([0-9.]+)"
    )
    for raw in text.splitlines():
        line = raw.strip("\n")
        m_tok = token_re.match(line)
        if m_tok is not None:
            cur_token = int(m_tok.group(1))
            metrics[cur_token] = {}
            continue
        m_line = line_re.match(line)
        if m_line is not None and cur_token is not None:
            kind = m_line.group(1)
            metrics[cur_token][kind] = {
                "cos_to_t1_mean": float(m_line.group(2)),
                "cos_to_t1_min": float(m_line.group(3)),
                "cos_to_prev_mean": float(m_line.group(4)),
                "cos_to_prev_min": float(m_line.group(5)),
            }
    if not metrics:
        raise ValueError("Could not parse drift metrics from output.")
    return {"token_to_step": token_map, "metrics": metrics}


def warn_if_diff(name: str, actual: float, expected: float, tol: float, *, tag: str = "Gemma4DiagWarn") -> None:
    delta = abs(actual - expected)
    if delta > tol:
        warnings.warn(
            f"[{tag}] {name} drifted: actual={actual:.6f} expected={expected:.6f} "
            f"abs_diff={delta:.6f} tol={tol:.6f}",
            stacklevel=2,
        )


def warn_if_token_mismatch(name: str, actual: int, expected: int, *, tag: str = "Gemma4DiagWarn") -> None:
    if actual != expected:
        warnings.warn(
            f"[{tag}] {name} changed: actual={actual} expected={expected}",
            stacklevel=2,
        )
