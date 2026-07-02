# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from vllm.engine.request_state import RequestState


class Sampler:
    """Vectorized temperature / top-k / top-p / multinomial sampling."""

    def sample(
        self,
        logits: torch.Tensor,
        requests: list[RequestState],
    ) -> list[int]:
        """Sample next tokens for a batch of requests.

        Applies temperature scaling, top-k filtering, top-p (nucleus) filtering,
        and multinomial sampling in a vectorized manner. Requests with
        ``temperature <= 1e-6`` use greedy argmax sampling.

        Args:
            logits: A 1-D or 2-D tensor of shape ``(batch_size, vocab_size)``
                containing raw logits for the current step.
            requests: A list of :class:`RequestState` objects, one per batch
                row, holding ``sampling_params`` and per-request RNG state.

        Returns:
            A list of integer token IDs, one for each request in the batch.

        Note:
            The input ``logits`` tensor may be mutated in-place (e.g., during
            temperature scaling and scatter-back from top-p/top-k filtering).
            Callers should pass a clone if they need to retain the original.
        """
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        device = logits.device
        logits = logits.float()
        batch_size, vocab_size = logits.shape

        greedy_flags = [
            float(getattr(req.sampling_params, "temperature", 0.0) or 0.0) <= 1e-6
            for req in requests
        ]
        if all(greedy_flags):
            return [int(torch.argmax(logits[i]).item()) for i in range(batch_size)]

        temps = torch.tensor(
            [
                float(getattr(req.sampling_params, "temperature", 0.0) or 0.0)
                for req in requests
            ],
            dtype=torch.float,
            device=device,
        ).unsqueeze(1)
        top_ks = torch.tensor(
            [int(getattr(req.sampling_params, "top_k", -1) or -1) for req in requests],
            dtype=torch.long,
            device=device,
        ).unsqueeze(1)
        top_ps = torch.tensor(
            [
                float(getattr(req.sampling_params, "top_p", 1.0) or 1.0)
                for req in requests
            ],
            dtype=torch.float,
            device=device,
        ).unsqueeze(1)

        non_greedy_mask = temps > 1e-6
        logits = torch.where(non_greedy_mask, logits / temps.clamp(min=1e-6), logits)

        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        col_indices = torch.arange(vocab_size, device=device).unsqueeze(0)
        top_k_mask = (col_indices >= top_ks) & (top_ks > 0)
        sorted_logits.masked_fill_(top_k_mask, float("-inf"))

        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(sorted_probs, dim=-1)
        sorted_remove = cumprobs > top_ps
        sorted_remove_shifted = torch.empty_like(sorted_remove)
        sorted_remove_shifted[:, 1:] = sorted_remove[:, :-1]
        sorted_remove_shifted[:, 0] = False
        sorted_logits.masked_fill_(sorted_remove_shifted, float("-inf"))

        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
        probs = torch.softmax(logits, dim=-1)

        next_tokens: list[int] = []
        for i, req in enumerate(requests):
            if greedy_flags[i]:
                next_tokens.append(int(torch.argmax(logits[i]).item()))
                continue
            row_probs = probs[i]
            psum = row_probs.sum()
            if not torch.isfinite(psum) or psum <= 0:
                next_tokens.append(int(torch.argmax(logits[i]).item()))
                continue
            generator = req.rng
            if generator is not None:
                sample = torch.multinomial(row_probs, 1, generator=generator)
            else:
                sample = torch.multinomial(row_probs, 1)
            next_tokens.append(int(sample.item()))
        return next_tokens
