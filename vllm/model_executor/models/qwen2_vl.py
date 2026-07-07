# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.multimodal_utils import replace_image_placeholders


def build_qwen2_vl_vision_tower(hf_config: Any) -> nn.Module:
    from transformers.models.qwen2_vl.modeling_qwen2_vl import (
        Qwen2VisionTransformerPretrainedModel,
        Qwen2VLVisionConfig,
    )

    vision_config = hf_config.vision_config
    if not isinstance(vision_config, Qwen2VLVisionConfig):
        values = (
            dict(vision_config)
            if isinstance(vision_config, dict)
            else vars(vision_config)
        )
        vision_config = Qwen2VLVisionConfig(**values)
    return Qwen2VisionTransformerPretrainedModel(vision_config)


def qwen2_vl_visual_state_dict_from_checkpoint(
    model_path: str | Path,
) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    root = Path(model_path)
    tensors: dict[str, torch.Tensor] = {}
    for shard in sorted(root.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            keys = handle.keys()
            for key in keys:
                if key.startswith("visual."):
                    tensors[key.removeprefix("visual.")] = handle.get_tensor(key)
    return tensors


class Qwen2VLForCausalLM(nn.Module):
    supports_multimodal = True

    def __init__(self, vllm_config: Any, prefix: str = "") -> None:
        super().__init__()
        del prefix
        hf_config = vllm_config.model_config.hf_config
        self.config = hf_config
        self.model = LlamaModel(
            hf_config,
            vllm_config.quant_config,
            "model",
        )
        self._install_mrope(hf_config)
        self.visual = build_qwen2_vl_vision_tower(hf_config)
        self.lm_head = LiteLinear(
            self.model.config.hidden_size,
            self.model.config.vocab_size,
            bias=False,
            prefix="lm_head",
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Any,
        attn_metadata: Any,
        pixel_values: torch.Tensor | None = None,
        multimodal_embeddings: torch.Tensor | None = None,
        lora_mapping: Any = None,
    ) -> torch.Tensor:
        del pixel_values
        image_token_id = int(attn_metadata.get("image_token_id", -1))
        image_token_count = int(attn_metadata.get("image_token_count", 0))
        hidden_inputs = self._merge_multimodal_embeddings(
            input_ids=input_ids,
            multimodal_embeddings=multimodal_embeddings,
            image_token_id=image_token_id,
            image_token_count=image_token_count,
        )
        image_grid_thw = attn_metadata.get("image_grid_thw")
        if image_grid_thw is not None:
            positions = self._qwen2_vl_positions(
                input_ids=input_ids,
                positions=positions,
                image_grid_thw=image_grid_thw,
                image_token_id=image_token_id,
            )
        elif positions is not None and positions.ndim == 2:
            positions = positions.unsqueeze(0).expand(3, -1, -1).contiguous()
        hidden_states = self.model(
            hidden_inputs,
            positions,
            kv_caches,
            attn_metadata,
            lora_mapping=lora_mapping,
        )
        if getattr(self.model.config, "tie_word_embeddings", False):
            return torch.nn.functional.linear(
                hidden_states[:, -1:, :],
                self.model.embed_tokens.weight,
            )
        return self.lm_head(hidden_states[:, -1:, :], lora_mapping)

    def _merge_multimodal_embeddings(
        self,
        *,
        input_ids: torch.Tensor,
        multimodal_embeddings: torch.Tensor | None,
        image_token_id: int = -1,
        image_token_count: int = 0,
    ) -> torch.Tensor:
        if input_ids.dtype == torch.long:
            hidden_states = self.model.embed_tokens(input_ids)
        else:
            hidden_states = input_ids
        if multimodal_embeddings is None:
            return hidden_states
        return replace_image_placeholders(
            input_ids=input_ids,
            text_embeddings=hidden_states,
            multimodal_embeddings=multimodal_embeddings,
            image_token_id=image_token_id,
            image_token_count=image_token_count,
        )

    def get_multimodal_embeddings(self, **kwargs: Any) -> torch.Tensor:
        pixel_values = kwargs.get("pixel_values")
        grid_thw = kwargs.get("image_grid_thw", kwargs.get("grid_thw"))
        if pixel_values is None:
            raise ValueError("Qwen2VL multimodal input requires pixel_values")
        if grid_thw is None:
            raise ValueError("Qwen2VL multimodal input requires image_grid_thw")
        try:
            visual_param = next(self.visual.parameters())
        except StopIteration:
            visual_param = None
        if visual_param is not None and visual_param.numel() > 0:
            pixel_values = pixel_values.to(device=visual_param.device)
            grid_thw = grid_thw.to(device=visual_param.device)
        output = self.visual(pixel_values, grid_thw=grid_thw)
        embeddings = output.pooler_output
        image_token_count = int(kwargs.get("image_token_count", 0) or 0)
        if image_token_count <= 0:
            return embeddings
        if embeddings.shape[0] < image_token_count:
            raise ValueError("Qwen2VL vision embeddings shorter than image_token_count")
        return embeddings[:image_token_count, :]

    def _install_mrope(self, hf_config: Any) -> None:
        rope_scaling = getattr(hf_config, "rope_scaling", None) or {}
        if not isinstance(rope_scaling, dict) or rope_scaling.get("type") != "mrope":
            return
        mrope_section = rope_scaling.get("mrope_section")
        if not mrope_section:
            return
        for layer in self.model.layers:
            head_dim = int(
                getattr(
                    self.model.config,
                    "head_dim",
                    layer.self_attn.head_dim,
                )
            )
            layer.rotary_emb = MRotaryEmbedding(
                head_dim,
                head_dim,
                self.model.config.max_position_embeddings,
                float(getattr(hf_config, "rope_theta", self.model.config.rope_theta)),
                is_neox_style=True,
                dtype=torch.float16,
                mrope_section=list(mrope_section),
                mrope_interleaved=True,
            )

    def _qwen2_vl_positions(
        self,
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        image_grid_thw: torch.Tensor,
        image_token_id: int,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("Qwen2VL input_ids must be 2D for mRoPE positions")
        if positions.ndim == 3:
            return positions
        output = positions.unsqueeze(0).expand(3, -1, -1).clone()
        merge = int(getattr(self.config.vision_config, "spatial_merge_size", 2) or 2)
        grids = image_grid_thw.to(device=positions.device, dtype=torch.long)
        grid_index = 0
        for batch_idx in range(input_ids.shape[0]):
            image_positions = (input_ids[batch_idx] == int(image_token_id)).nonzero(
                as_tuple=True
            )[0]
            offset = 0
            while offset < image_positions.numel():
                if grid_index >= grids.shape[0]:
                    raise ValueError(
                        "image_grid_thw count is shorter than image tokens"
                    )
                t, h, w = [int(x) for x in grids[grid_index].tolist()]
                hh = h // merge
                ww = w // merge
                count = t * hh * ww
                if offset + count > image_positions.numel():
                    raise ValueError("image_grid_thw token count exceeds image tokens")
                media_pos = image_positions[offset : offset + count]
                base = int(positions[batch_idx, int(media_pos[0])].item())
                coords = self._qwen2_vl_image_positions(t, hh, ww, positions.device)
                output[:, batch_idx, media_pos] = coords + base
                tail = media_pos[-1].item() + 1
                if tail < positions.shape[1]:
                    next_pos = int(output[:, batch_idx, media_pos].max().item()) + 1
                    delta = next_pos - int(positions[batch_idx, tail].item())
                    output[:, batch_idx, tail:] = positions[batch_idx, tail:] + delta
                offset += count
                grid_index += 1
            if offset != image_positions.numel():
                raise ValueError("unused Qwen2VL image tokens")
        if grid_index != grids.shape[0]:
            raise ValueError("image_grid_thw count exceeds image tokens")
        return output

    @staticmethod
    def _qwen2_vl_image_positions(
        t: int,
        h: int,
        w: int,
        device: torch.device,
    ) -> torch.Tensor:
        t_index = torch.arange(t, device=device).view(-1, 1).expand(-1, h * w)
        h_index = torch.arange(h, device=device).view(1, -1, 1).expand(t, -1, w)
        w_index = torch.arange(w, device=device).view(1, 1, -1).expand(t, h, -1)
        return torch.stack(
            [t_index.flatten(), h_index.flatten(), w_index.flatten()],
            dim=0,
        )


class Qwen2VLForConditionalGeneration(Qwen2VLForCausalLM):
    pass
