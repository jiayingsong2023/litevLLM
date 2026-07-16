# SPDX-License-Identifier: Apache-2.0
"""Audit Gemma4 12B checkpoint projections for the exact batch contract."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "models/gemma-4-12B-it-AWQ-INT4"
_SUFFIXES = (
    ".self_attn.q_proj",
    ".self_attn.k_proj",
    ".self_attn.v_proj",
    ".self_attn.o_proj",
    ".mlp.gate_proj",
    ".mlp.up_proj",
    ".mlp.down_proj",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _role(name: str) -> str | None:
    for suffix in _SUFFIXES:
        if name.endswith(suffix):
            return suffix.removeprefix(".self_attn.").removeprefix(".mlp.")
    return None


def _checkpoint_layers(model: Any) -> list[tuple[str, Any]]:
    from vllm.model_executor.layers.lite_linear import LiteLinear

    selected: dict[tuple[str, tuple[int, ...], tuple[int, ...]], tuple[str, Any]] = {}
    for name, layer in model.named_modules():
        role = _role(name)
        qweight = getattr(layer, "qweight", None)
        scales = getattr(layer, "scales", None)
        if (
            role is None
            or not isinstance(layer, LiteLinear)
            or qweight is None
            or scales is None
            or qweight.numel() <= 1
        ):
            continue
        key = (role, tuple(qweight.shape), tuple(scales.shape))
        selected.setdefault(key, (name, layer))
    missing = {suffix.rsplit(".", 1)[-1] for suffix in _SUFFIXES} - {
        _role(name) for name, _ in selected.values()
    }
    if missing:
        raise RuntimeError(
            f"checkpoint lacks required projection roles: {sorted(missing)}"
        )
    return list(selected.values())


def _median_ms(fn: Any, repetitions: int) -> float:
    import torch

    fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repetitions):
        torch.cuda.synchronize()
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    return float(statistics.median(samples))


def _run_layer(layer: Any, x: Any, config: Any) -> Any:
    return layer(x, inf_config=config)


def main() -> None:
    args = parse_args()
    if args.repetitions < 1:
        raise ValueError("--repetitions must be positive")

    import torch

    from vllm import LLM
    from vllm.model_executor.layers.quantization.tensor import (
        get_awq_runtime_prefix_stats,
        reset_awq_runtime_stats,
    )

    llm = LLM(args.model, max_num_seqs=4, max_model_len=256)
    # The row-exact kernel must be compared with the exact M=1 production
    # path.  Replacing the full runtime policy changes M=1 dispatch and turns
    # this into an invalid comparison.
    config = llm.engine.inf_config
    config.kernel_policy["awq_rows_exact_msmall"] = True
    rows: list[dict[str, object]] = []
    for name, layer in _checkpoint_layers(llm.model):
        for m in (2, 4):
            torch.manual_seed(1000 + m + len(rows))
            x = torch.randn(
                (m, int(layer.input_size)), device="cuda", dtype=torch.bfloat16
            )
            reset_awq_runtime_stats()
            batched = _run_layer(layer, x, config)
            independent = torch.cat(
                [_run_layer(layer, x[row : row + 1], config) for row in range(m)],
                dim=0,
            )
            torch.cuda.synchronize()
            prefix_stats = get_awq_runtime_prefix_stats()
            quant_weight = getattr(layer, "_quant_weight", None)
            exact_prefix = str(getattr(quant_weight, "prefix", layer.prefix))
            exact_dispatches = int(
                prefix_stats.get(exact_prefix, {}).get(
                    "rows_exact_msmall_success", 0
                )
            )
            rows.append(
                {
                    "layer": name,
                    "shape": list(layer.qweight.shape),
                    "m": m,
                    "torch_equal": bool(torch.equal(batched, independent)),
                    "row_exact_dispatches": exact_dispatches,
                    "m1_total_ms": _median_ms(
                        lambda layer=layer, x=x, m=m: torch.cat(
                            [
                                _run_layer(layer, x[row : row + 1], config)
                                for row in range(m)
                            ],
                            dim=0,
                        ),
                        args.repetitions,
                    ),
                    "batched_ms": _median_ms(
                        lambda layer=layer, x=x: _run_layer(layer, x, config),
                        args.repetitions,
                    ),
                }
            )

    for row in rows:
        m1_total_ms = float(row["m1_total_ms"])
        batched_ms = float(row["batched_ms"])
        row["speedup"] = m1_total_ms / batched_ms if batched_ms else 0.0
        row["performance_passed"] = batched_ms < m1_total_ms
    result = {
        "model": args.model,
        "repetitions": args.repetitions,
        "rows": rows,
        "passed": all(
            bool(row["torch_equal"])
            and int(row["row_exact_dispatches"]) > 0
            and bool(row["performance_passed"])
            for row in rows
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise RuntimeError("row-exact checkpoint audit gate failed")


if __name__ == "__main__":
    main()
