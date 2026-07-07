# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch


def replace_image_placeholders(
    *,
    input_ids: torch.Tensor,
    text_embeddings: torch.Tensor,
    multimodal_embeddings: torch.Tensor,
    image_token_id: int,
    image_token_count: int,
) -> torch.Tensor:
    if image_token_count <= 0:
        return text_embeddings
    image_mask = input_ids.eq(int(image_token_id))
    if int(image_mask.sum().item()) != int(image_token_count):
        raise ValueError("image placeholder count does not match image_token_count")
    if multimodal_embeddings.dim() == 2:
        multimodal_embeddings = multimodal_embeddings.unsqueeze(0)
    image_embeddings = multimodal_embeddings.reshape(-1, text_embeddings.shape[-1])
    if image_embeddings.shape[0] != int(image_token_count):
        raise ValueError("image embedding count does not match image_token_count")
    output = text_embeddings.clone()
    output[image_mask] = image_embeddings.to(dtype=output.dtype, device=output.device)
    return output
