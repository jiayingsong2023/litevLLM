# SPDX-License-Identifier: Apache-2.0
"""
Layer-wise activation audit for Lite causal-LM style modules.

Attach forward hooks on embedding, each decoder layer, final norm, and optional lm_head.
Useful to locate the first layer where hidden states become NaN/Inf, collapse to ~0, or explode.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable

import torch
import torch.nn as nn


def _first_tensor_output(out):
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
        return out[0]
    return None


def _tensor_stats_line(
    name: str,
    pass_id: int,
    t: torch.Tensor,
) -> str:
    t = t.detach()
    shape = tuple(t.shape)
    flat = t.float().reshape(-1)
    n = flat.numel()
    if n == 0:
        return f"[Activation] pass={pass_id} {name} shape={shape} EMPTY"
    n_nan = int(torch.isnan(flat).sum().item())
    n_inf = int(torch.isinf(flat).sum().item())
    finite = torch.isfinite(flat)
    n_fin = int(finite.sum().item())
    if n_fin == 0:
        return (
            f"[Activation] pass={pass_id} {name} shape={shape} "
            f"mean=nan std=nan min=nan max=nan finite=0/{n} nan={n_nan} inf={n_inf}"
        )
    vals = flat[finite]
    return (
        f"[Activation] pass={pass_id} {name} shape={shape} "
        f"mean={vals.mean().item():.6g} std={vals.std(unbiased=False).item():.6g} "
        f"min={vals.min().item():.6g} max={vals.max().item():.6g} "
        f"finite={n_fin}/{n} nan={n_nan} inf={n_inf}"
    )


def resolve_lite_backbone(lite_model: nn.Module) -> nn.Module | None:
    """
    Find inner backbone holding embed_tokens + layers (e.g. LlamaModel / Qwen2Model under ForCausalLM).
    """
    inner = getattr(lite_model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner
    if hasattr(lite_model, "layers"):
        return lite_model
    return None


class LiteActivationSniffer:
    """
    Register hooks on a Lite HF-like model for per-forward activation statistics.

    Args:
        lite_model: Top module (e.g. Qwen3_5ForConditionalGeneration, LlamaForCausalLM wrapper).
        print_fn: Defaults to builtins.print.
        include_embeddings: Hook embed_tokens output.
        include_final_norm: Hook final RMSNorm / LayerNorm before lm_head.
        include_lm_head: Hook lm_head output (logits).
        max_passes: If set, stop printing after this many top-level forward calls (decode can be noisy).
    """

    def __init__(
        self,
        lite_model: nn.Module,
        print_fn: Callable[[str], None] | None = None,
        *,
        include_embeddings: bool = True,
        include_final_norm: bool = True,
        include_lm_head: bool = True,
        max_passes: int | None = None,
    ) -> None:
        self.lite_model = lite_model
        self.print_fn = print_fn or print
        self.include_embeddings = include_embeddings
        self.include_final_norm = include_final_norm
        self.include_lm_head = include_lm_head
        self.max_passes = max_passes

        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._pass_id = 0
        self._enabled = False

    def attach(self) -> None:
        if self._handles:
            self.detach()
        backbone = resolve_lite_backbone(self.lite_model)
        if backbone is None:
            self.print_fn(
                "[Activation] Could not resolve backbone (no .model.layers); no hooks attached."
            )
            return

        def _root_pre(_mod, _inp):
            self._pass_id += 1

        self._handles.append(self.lite_model.register_forward_pre_hook(_root_pre))

        def _make_hook(tag: str):
            def _hook(_mod, _inp, out):
                if self.max_passes is not None and self._pass_id > self.max_passes:
                    return
                t = _first_tensor_output(out)
                if t is None:
                    return
                self.print_fn(_tensor_stats_line(tag, self._pass_id, t))

            return _hook

        if self.include_embeddings and hasattr(backbone, "embed_tokens"):
            emb = backbone.embed_tokens
            self._handles.append(emb.register_forward_hook(_make_hook("embed_tokens")))

        layers = getattr(backbone, "layers", None)
        if layers is not None:
            for i, layer in enumerate(layers):
                self._handles.append(
                    layer.register_forward_hook(_make_hook(f"layer.{i:02d}.out"))
                )

        if (
            self.include_final_norm
            and hasattr(backbone, "norm")
            and backbone.norm is not None
        ):
            self._handles.append(
                backbone.norm.register_forward_hook(_make_hook("final_norm"))
            )

        if (
            self.include_lm_head
            and hasattr(self.lite_model, "lm_head")
            and self.lite_model.lm_head is not None
        ):
            self._handles.append(
                self.lite_model.lm_head.register_forward_hook(
                    _make_hook("lm_head.logits")
                )
            )

        self._enabled = True
        self.print_fn(
            f"[Activation] Sniffer attached: backbone={type(backbone).__name__}, "
            f"layers={len(layers) if layers is not None else 0}, max_passes={self.max_passes}"
        )

    def detach(self) -> None:
        for h in self._handles:
            with contextlib.suppress(Exception):
                h.remove()
        self._handles.clear()
        if self._enabled:
            self.print_fn("[Activation] Sniffer detached.")
        self._enabled = False

    def __enter__(self) -> LiteActivationSniffer:
        self.attach()
        return self

    def __exit__(self, *args) -> None:
        self.detach()
