import torch


def fused_moe(hidden_states, w1, w2, gating_output, topk, renormalize=True):
    """
    MoE dispatcher for the maintained FP16/BF16 execution path.
    """
    M, K = hidden_states.shape
    E = w1.shape[0]

    # 1. Routing
    routing_weights = torch.softmax(gating_output, dim=-1)
    topk_weights, topk_ids = torch.topk(routing_weights, topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    output = torch.zeros_like(hidden_states)

    # 2. Strategy Selection
    # For massive models or large tokens, use Serialized Dispatch to ensure stability.
    use_serialized_mode = E >= 256 or K > 4096 or M > 128

    if not use_serialized_mode:
        flattened_ids = topk_ids.view(-1)
        flattened_weights = topk_weights.view(-1)
        token_indices = torch.arange(M, device=hidden_states.device).repeat_interleave(
            topk
        )

        sorted_ids, sorting_indices = torch.sort(flattened_ids)
        sorted_token_indices = token_indices[sorting_indices]
        sorted_weights = flattened_weights[sorting_indices]

        unique_ids, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
        unique_ids_list = unique_ids.tolist()
        counts_list = counts.tolist()

        curr_offset = 0
        for i, expert_idx in enumerate(unique_ids_list):
            count = counts_list[i]
            start, end = curr_offset, curr_offset + count
            curr_offset += count

            group_token_indices = sorted_token_indices[start:end]
            group_weights = sorted_weights[start:end].unsqueeze(-1)
            tokens = hidden_states.index_select(0, group_token_indices)

            res = torch.nn.functional.linear(tokens, w1[expert_idx])
            if res.shape[-1] == 2 * w2.shape[-1]:
                d = res.shape[-1] // 2
                res = torch.nn.functional.silu(res[:, :d]) * res[:, d:]
            else:
                res = torch.nn.functional.silu(res)
            res = torch.nn.functional.linear(res, w2[expert_idx])

            output.index_add_(0, group_token_indices, res * group_weights)
    else:
        # Serialized path for stability
        for m in range(M):
            token_expert_ids = topk_ids[m]
            token_expert_weights = topk_weights[m]
            token_hidden = hidden_states[m : m + 1]
            token_output = torch.zeros_like(token_hidden)
            for k in range(topk):
                expert_idx = token_expert_ids[k].item()
                weight = token_expert_weights[k].item()
                res = torch.nn.functional.linear(token_hidden, w1[expert_idx])
                if res.shape[-1] == 2 * w2.shape[-1]:
                    d = res.shape[-1] // 2
                    res = torch.nn.functional.silu(res[:, :d]) * res[:, d:]
                else:
                    res = torch.nn.functional.silu(res)
                res = torch.nn.functional.linear(res, w2[expert_idx])
                token_output += res * weight
            output[m] = token_output

    return output


__all__ = ["fused_moe"]
