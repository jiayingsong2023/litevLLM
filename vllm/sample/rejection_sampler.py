# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from dataclasses import replace

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1_outputs import LogprobsLists, LogprobsTensors, SamplerOutput
from vllm.sample.metadata import SamplingMetadata
from vllm.sample.ops.bad_words import apply_bad_words_with_drafts
from vllm.sample.ops.penalties import apply_all_penalties
from vllm.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.sample.sampler import Sampler
from vllm.spec_decode.metadata import SpecDecodeMetadata

logger = init_logger(__name__)

PLACEHOLDER_TOKEN_ID: tl.constexpr = -1
GREEDY_TEMPERATURE: tl.constexpr = 0
# Maximum number of speculative draft tokens allowed per request in a single
# step. This value is chosen to be large enough to handle typical use cases.
MAX_SPEC_LEN = 128

class RejectionSampler(nn.Module):

    def __init__(self, sampler: Sampler):
        super().__init__()
        self.sampler = sampler
        logprobs_mode = self.sampler.logprobs_mode
        self.is_processed_logprobs_mode = logprobs_mode.startswith("processed")
        self.is_logits_logprobs_mode = logprobs_mode.endswith("logits")

    def forward(
        self,
        metadata: SpecDecodeMetadata,
        # [num_tokens, vocab_size]
        draft_probs: torch.Tensor | None,
        # [num_tokens + batch_size, vocab_size]
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        assert metadata.max_spec_len <= MAX_SPEC_LEN

        bonus_logits_indices = metadata.bonus_logits_indices
        target_logits_indices = metadata.target_logits_indices

        # When indexing with a tensor (bonus_logits_indices), PyTorch
        # creates a new tensor with separate storage from the original
        # logits tensor. This means any in-place operations on bonus_logits
        # won't affect the original logits tensor.
        assert logits is not None
        bonus_logits = logits[bonus_logits_indices]
        bonus_sampler_output = self.sampler(
            logits=bonus_logits,
            sampling_metadata=replace(
                sampling_metadata,
                max_num_logprobs=-1,
            ),
            predict_bonus_token=True,
            # Override the logprobs mode to return logits because they are
            # needed later to compute the accepted token logprobs.
            logprobs_mode_override="processed_logits"
            if self.is_processed_logprobs_mode
            else "raw_logits",
        )
        bonus_token_ids = bonus_sampler_output.sampled_token_ids

        # Just like `bonus_logits`, `target_logits` is a new tensor with
        # separate storage from the original `logits` tensor. Therefore,
        # it is safe to update `target_logits` in place.
        raw_target_logits = logits[target_logits_indices]
        # Use float32 for the target_logits.
        raw_target_logits = raw_target_logits.to(torch.float32)
        target_logits = raw_target_logits
        if not self.is_processed_logprobs_mode:
            # Clone raw_target_logits before applying processors to preserve
            # the original raw logits for logprobs computation, since
            # apply_logits_processors modifies the tensor in-place.
            target_logits = target_logits.clone()
        target_logits = self.apply_logits_processors(
            target_logits, sampling_metadata, metadata
        )
        # [num_tokens, vocab_size]
        # NOTE(woosuk): `target_logits` can be updated in place inside the
        # `apply_sampling_constraints` function.
        target_logits = apply_sampling_constraints(
            target_logits,
            metadata.cu_num_draft_tokens,
            sampling_metadata,
        )

        output_token_ids = rejection_sample(
            metadata.draft_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
            metadata.cu_num_draft_tokens,
            draft_probs,
            target_logits,
            bonus_token_ids,
            sampling_metadata,
        )

        logprobs_tensors = None
        if sampling_metadata.max_num_logprobs is not None:
            logprobs_tensors = self._get_logprobs_tensors(
                sampling_metadata.max_num_logprobs,
                metadata,
                logits,
                target_logits if self.is_processed_logprobs_mode else raw_target_logits,
                bonus_sampler_output.logprobs_tensors.logprobs,
                output_token_ids,
            )

        return SamplerOutput(
            sampled_token_ids=output_token_ids,
            logprobs_tensors=logprobs_tensors,
        )

    def _get_logprobs_tensors(
        self,
        max_num_logprobs: int,
        metadata: SpecDecodeMetadata,
        logits: torch.Tensor,
        target_logits: torch.Tensor,
        bonus_logits: torch.Tensor,
        sampled_token_ids: torch.Tensor,
    ) -> LogprobsTensors:
        cu_num_sampled_tokens = torch.zeros_like(metadata.cu_num_sampled_tokens)
        cu_num_sampled_tokens[1:] = metadata.cu_num_sampled_tokens[:-1]

        # Collect target and bonus logits.
        bonus_logits_indices = metadata.bonus_logits_indices
        target_logits_indices = metadata.target_logits_indices
        final_logits = torch.zeros_like(logits, dtype=torch.float32)
        final_logits[target_logits_indices] = target_logits.to(torch.float32)
        final_logits[bonus_logits_indices] = bonus_logits.to(torch.float32)

        # NOTE: To avoid cpu-gpu synchronization, we now simply compute indices for
        # all draft tokens, including the rejected ones. The rejected tokens will
        # be filtered out in the `parse_output`.
        logit_start_indices = cu_num_sampled_tokens
        offsets = torch.arange(
            sampled_token_ids.shape[-1],
            device=logit_start_indices.device,
            dtype=logit_start_indices.dtype,
        )
        accepted_logit_indices = (
            logit_start_indices.unsqueeze(1) + offsets.unsqueeze(0)
        ).flatten()
        accepted_logit_indices.clamp_(max=final_logits.shape[0] - 1)
        accepted_tokens = sampled_token_ids.clone().flatten()
        # we replace rejected token ids with 0 to avoid gather_logprobs error
        accepted_tokens[accepted_tokens == PLACEHOLDER_TOKEN_ID] = 0

        # Compute logprobs for accepted tokens.
        accepted_logits = final_logits[accepted_logit_indices]
        accepted_logprobs = (
            accepted_logits
            if self.is_logits_logprobs_mode
            else self.sampler.compute_logprobs(accepted_logits)
        )
        return self.sampler.gather_logprobs(
            accepted_logprobs,
            max_num_logprobs,
            accepted_tokens.to(torch.int64),
        )

    @staticmethod
    def parse_output(
        output_token_ids: torch.Tensor,
        vocab_size: int,
        discard_req_indices: Sequence[int] = (),
        logprobs_tensors: LogprobsTensors | None = None,
    ) -> tuple[list[list[int]], LogprobsLists | None]:
        output_token_ids_np = output_token_ids.cpu().numpy()
        # Create mask for valid tokens.
        valid_mask = (output_token_ids_np != PLACEHOLDER_TOKEN_ID) & (
            output_token_ids_np < vocab_size
        )
        output_logprobs = None
        if logprobs_tensors is not None:
            cu_num_tokens = [0] + valid_mask.sum(axis=1).cumsum().tolist()
            filtered_tensors = logprobs_tensors.filter(valid_mask.flatten())
            output_logprobs = filtered_tensors.tolists(cu_num_tokens)

        if len(discard_req_indices) > 0:
            valid_mask[discard_req_indices] = False
        outputs = [
            row[valid_mask[i]].tolist() for i, row in enumerate(output_token_ids_np)
        ]
        return outputs, output_logprobs

    def apply_logits_processors(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        metadata: SpecDecodeMetadata,
    ) -> torch.Tensor:
        has_penalties = not sampling_metadata.no_penalties
        any_penalties_or_bad_words = (
            sampling_metadata.bad_words_token_ids or has_penalties
        )

        output_token_ids = sampling_metadata.output_token_ids
        if any_penalties_or_bad_words:
            output_token_ids = self._combine_outputs_with_spec_tokens(
                output_token_ids,
                sampling_metadata.spec_token_ids,
            )

        # Calculate indices of target logits.
        if sampling_metadata.allowed_token_ids_mask is not None or has_penalties:
            num_requests = len(sampling_metadata.output_token_ids)
            num_draft_tokens = torch.tensor(metadata.num_draft_tokens, device="cpu")
            original_indices = torch.arange(num_requests, device="cpu")
            repeat_indices_cpu = original_indices.repeat_interleave(num_draft_tokens)
            repeat_indices = repeat_indices_cpu.to(
                device=logits.device, non_blocking=True
            )
            logits = self.apply_penalties(
                logits, sampling_metadata, metadata, repeat_indices, output_token_ids
            )

            # Apply allowed token ids.
            if sampling_metadata.allowed_token_ids_mask is not None:
                token_mask = sampling_metadata.allowed_token_ids_mask[repeat_indices]
                logits.masked_fill_(token_mask, float("-inf"))

        # Apply bad words exclusion.
        if bad_words_token_ids := sampling_metadata.bad_words_token_ids:
            apply_bad_words_with_drafts(
                logits, bad_words_token_ids, output_token_ids, metadata.num_draft_tokens
            )

        return logits

    @staticmethod
    def apply_penalties(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        metadata: SpecDecodeMetadata,
        repeat_indices: torch.Tensor,
        output_token_ids: list[list[int]],
    ) -> torch.Tensor:
        if sampling_metadata.no_penalties:
            return logits

        assert sampling_metadata.prompt_token_ids is not None

        prompt_token_ids = sampling_metadata.prompt_token_ids[repeat_indices]
        presence_penalties = sampling_metadata.presence_penalties[repeat_indices]
        frequency_penalties = sampling_metadata.frequency_penalties[repeat_indices]
        repetition_penalties = sampling_metadata.repetition_penalties[repeat_indices]

        logits = apply_all_penalties(
            logits,
            prompt_token_ids,
            presence_penalties,
            frequency_penalties,
            repetition_penalties,
            output_token_ids,
        )
        return logits

    @staticmethod
    def _combine_outputs_with_spec_tokens(
        output_token_ids: list[list[int]],
        spec_token_ids: list[list[int]] | None = None,
    ) -> list[list[int]]:
        if spec_token_ids is None:
            return output_token_ids

        result = []
        for out, spec in zip(output_token_ids, spec_token_ids):
            if len(spec) == 0:
                continue
            result.append(out)
            for i in range(len(spec) - 1):
                result.append([*result[-1], spec[i]])
        return result

def rejection_sample(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    max_spec_len: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: torch.Tensor | None,
    # [num_tokens, vocab_size]
    target_logits: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    assert draft_token_ids.ndim == 1
    assert draft_probs is None or draft_probs.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    assert target_logits.ndim == 2

    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    vocab_size = target_logits.shape[-1]
    device = target_logits.device
    assert draft_token_ids.is_contiguous()
    assert draft_probs is None or draft_probs.is_contiguous()
    assert bonus_token_ids.is_contiguous()
    assert target_logits.shape == (num_tokens, vocab_size)

    # Create output buffer.
    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )

    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE
    if not sampling_metadata.all_random:
        # Rejection sampling for greedy sampling requests.
        target_argmax = target_logits.argmax(dim=-1)
        rejection_greedy_sample_kernel[(batch_size,)](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            is_greedy,
            max_spec_len,
        )
        if sampling_metadata.all_greedy:
            return output_token_ids

    # Compute probability distribution from target logits.
    target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
    assert target_probs.is_contiguous()

    # Generate uniform probabilities for rejection sampling.
    # [num_tokens]
    uniform_probs = generate_uniform_probs(
        num_tokens,
        num_draft_tokens,
        sampling_metadata.generators,
        device,
    )

    # Sample recovered tokens for each position.
    # [num_tokens]
    recovered_token_ids = sample_recovered_tokens(
        max_spec_len,
        num_draft_tokens,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        sampling_metadata,
        device,
    )

    # Rejection sampling for random sampling requests.
    rejection_random_sample_kernel[(batch_size,)](
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
        NO_DRAFT_PROBS=draft_probs is None,
    )
    return output_token_ids

def apply_sampling_constraints(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    assert logits.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    if sampling_metadata.all_greedy:
        return logits

    num_tokens = logits.shape[0]
    temperature = expand_batch_to_tokens(
        sampling_metadata.temperature,
        cu_num_draft_tokens,
        num_tokens,
        replace_from=GREEDY_TEMPERATURE,
        replace_to=1,
    )
    # NOTE(woosuk): Update `logits` in place to avoid allocating a new tensor.
    logits.div_(temperature.unsqueeze(-1))

    # Get expanded top_k and top_p tensors.
    top_k = None
    if sampling_metadata.top_k is not None:
        top_k = expand_batch_to_tokens(
            sampling_metadata.top_k,
            cu_num_draft_tokens,
            num_tokens,
        )
    top_p = None
    if sampling_metadata.top_p is not None:
        top_p = expand_batch_to_tokens(
            sampling_metadata.top_p,
            cu_num_draft_tokens,
            num_tokens,
        )

    # NOTE(woosuk): `apply_top_k_top_p` uses sorting to calculate the mask,
    # which is slow for large vocab sizes. This may cause performance issues.
    return apply_top_k_top_p(logits, top_k, top_p)

def expand_batch_to_tokens(
    x: torch.Tensor,  # [batch_size]
    cu_num_tokens: torch.Tensor,  # [batch_size]
    num_tokens: int,
    replace_from: int = 0,
    replace_to: int = 0,
) -> torch.Tensor:
    batch_size = x.shape[0]
    assert cu_num_tokens.shape[0] == batch_size
    expanded_x = x.new_empty(num_tokens)
    expand_kernel[(batch_size,)](
        expanded_x,
        x,
        cu_num_tokens,
        replace_from,
        replace_to,
        MAX_NUM_TOKENS=MAX_SPEC_LEN,  # To avoid recompilation.
    )
    return expanded_x

def generate_uniform_probs(
    num_tokens: int,
    num_draft_tokens: list[int],
    generators: dict[int, torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    # NOTE(woosuk): We deliberately use float64 instead of float32 here
    # because when using float32, there's a non-negligible chance that
    # uniform_prob is sampled to be exact 0.0 as reported in
    # https://github.com/pytorch/pytorch/issues/16706. Using float64
    # mitigates the issue.
    uniform_probs = torch.rand(
        (num_tokens,),
        dtype=torch.float64,
        device=device,
    )
    start_idx = 0
    for req_idx, n in enumerate(num_draft_tokens):
        # Do not generate random numbers for requests with no draft tokens.
        # This can be important for reproducibility.
        if n == 0:
            continue
        end_idx = start_idx + n
        generator = generators.get(req_idx)
        if generator is not None:
            uniform_probs[start_idx:end_idx].uniform_(generator=generator)
        start_idx = end_idx
    return uniform_probs

def sample_recovered_tokens(
    max_spec_len: int,
    num_draft_tokens: list[int],
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: torch.Tensor | None,
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    device: torch.device,
) -> torch.Tensor:
    # NOTE(woosuk): Create only one distribution for each request.
    batch_size = len(num_draft_tokens)
    vocab_size = target_probs.shape[-1]
    q = torch.empty(
        (batch_size, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    q.exponential_()
    for i, generator in sampling_metadata.generators.items():
        # Do not generate random numbers for requests with no draft tokens.
        # This can be important for reproducibility.
        if num_draft_tokens[i] > 0:
            q[i].exponential_(generator=generator)

    recovered_token_ids = torch.empty_like(draft_token_ids)
    sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
        recovered_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        q,
        vocab_size,
        triton.next_power_of_2(vocab_size),
        NO_DRAFT_PROBS=draft_probs is None,
    )
    return recovered_token_ids

# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_greedy_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    is_greedy_ptr,  # [batch_size] or None
    max_spec_len,
):
    req_idx = tl.program_id(0)
    # FIXME(woosuk): Because is_greedy_ptr is not None at profiling run,
    # re-compilation may happen during runtime when is_greedy_ptr is None.
    is_greedy = True if is_greedy_ptr is None else tl.load(is_greedy_ptr + req_idx)
    if not is_greedy:
        # Early exit for non-greedy sampling requests.
        return

    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            target_argmax_id = tl.load(target_argmax_ptr + start_idx + pos)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                target_argmax_id,
            )
            if draft_token_id != target_argmax_id:
                # Reject.
                rejected = True

    if not rejected:
        # If all tokens are accepted, append the bonus token.
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )

# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_random_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    is_greedy_ptr,  # [batch_size]
    max_spec_len,
    vocab_size,
    NO_DRAFT_PROBS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    is_greedy = tl.load(is_greedy_ptr + req_idx)
    if is_greedy:
        # Early exit for greedy sampling requests.
        return

    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            if NO_DRAFT_PROBS:
                draft_prob = 1
            else:
                draft_prob = tl.load(
                    draft_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id
                )
            target_prob = tl.load(
                target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id
            )
            uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
            # NOTE(woosuk): While the draft probability should never be 0,
            # we check it to avoid NaNs. If it happens to be 0, we reject.
            if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
                # Accept.
                token_id = draft_token_id
            else:
                # Reject. Use recovered token.
                rejected = True
                token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id
            )

    if not rejected:
        # If all tokens are accepted, append the bonus token.
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )

# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["replace_from", "replace_to"])
def expand_kernel(
    output_ptr,  # [num_tokens]
    input_ptr,  # [batch_size]
    cu_num_tokens_ptr,  # [batch_size]
    replace_from,
    replace_to,
    MAX_NUM_TOKENS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    if req_idx == 0:  # noqa: SIM108
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_tokens_ptr + req_idx)
    num_tokens = end_idx - start_idx

    src_val = tl.load(input_ptr + req_idx)
    src_val = tl.where(src_val == replace_from, replace_to, src_val)
    offset = tl.arange(0, MAX_NUM_TOKENS)
    tl.store(output_ptr + start_idx + offset, src_val, mask=offset < num_tokens)

@triton.jit
def sample_recovered_tokens_kernel(
    output_token_ids_ptr,  # [num_tokens]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    q_ptr,  # [batch_size, vocab_size]
    vocab_size,
    PADDED_VOCAB_SIZE: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    # Early exit for out-of-range positions.
    pos = tl.program_id(1)
    if pos >= num_draft_tokens:
        return

    vocab_offset = tl.arange(0, PADDED_VOCAB_SIZE)
    if NO_DRAFT_PROBS:
        draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
        prob = tl.load(
            target_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset,
            mask=((vocab_offset < vocab_size) & (vocab_offset != draft_token_id)),
            other=0,
        )
    else:
        draft_prob = tl.load(
            draft_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset,
            mask=vocab_offset < vocab_size,
            other=0,
        )
        target_prob = tl.load(
            target_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset,
            mask=vocab_offset < vocab_size,
            other=0,
        )
        prob = tl.maximum(target_prob - draft_prob, 0)
        # NOTE(woosuk): We don't need `prob = prob / tl.sum(prob)` here because
        # `tl.argmax` will select the maximum value.

    q = tl.load(
        q_ptr + req_idx * vocab_size + vocab_offset,
        mask=vocab_offset < vocab_size,
        other=float("-inf"),
    )
    recovered_id = tl.argmax(prob / q, axis=-1)
    tl.store(output_token_ids_ptr + start_idx + pos, recovered_id)
