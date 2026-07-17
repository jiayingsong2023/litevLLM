# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import base64
import io
import math
import urllib.request
from typing import Any
from urllib.parse import urlparse

import torch
from PIL import Image

from vllm.engine.request_state import RequestState
from vllm.model_executor.models.interfaces import supports_multimodal


class NullMultiModalProcessor:
    """No-op multimodal processor for runtimes that reject multimodal input."""

    def prepare_request(self, request: RequestState) -> None:
        del request

    def build_prefill_inputs(self, req_dicts: list[RequestState]) -> dict[str, Any]:
        del req_dicts
        return {}

    def get_multimodal_embeddings(
        self,
        mm_inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        del mm_inputs
        return None

    def stats(self) -> dict[str, Any]:
        return {}

    def reset_stats(self) -> None:
        return None


class LiteMultiModalProcessor:
    """Minimal multimodal preprocessor for Lite runtime.

    Current scope supports single-request image batches, including multiple
    images attached to one request. The processor converts each image into a
    deterministic pixel tensor and aggregates image embeddings back to a
    request-level conditioning vector.
    """

    def __init__(
        self,
        *,
        model: Any,
        device: torch.device,
    ) -> None:
        self.model = model
        self.device = device
        self.supports_multimodal = supports_multimodal(model)
        self._qwen2_vl_image_processor: Any | None = None
        self.prepared_requests = 0
        self.prepared_images = 0
        self.prepare_failures = 0
        self.embedding_requests = 0
        self.embeddings_computed = 0
        self.embedding_feature_dim = 0

    def prepare_before_tokenize(
        self,
        prompt: str,
        multi_modal_data: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        if not self.supports_multimodal:
            raise ValueError("model does not support multimodal inputs")
        images = multi_modal_data.get("image")
        if not isinstance(images, list) or not images:
            raise ValueError("multi_modal_data.image must be a non-empty list")
        prompt_image_token = self._select_prompt_image_token(prompt, len(images))
        model_image_token = self._image_token()
        if prompt.count(prompt_image_token) != len(images):
            raise ValueError(
                f"prompt must contain one {prompt_image_token} placeholder per image"
            )
        prepared_images = []
        image_token_counts = []
        prompt_parts = prompt.split(prompt_image_token)
        expanded_parts = [prompt_parts[0]]
        for idx, item in enumerate(images):
            if not isinstance(item, dict) or not isinstance(item.get("image"), str):
                raise ValueError(
                    "multi_modal_data.image item must contain string image url"
                )
            image = self._load_image(item["image"])
            if self._is_qwen2_vl():
                prepared = self._prepare_qwen2_vl_image(image)
                image_token_count = int(prepared["image_token_count"])
            else:
                prepared = self._prepare_gemma4_image(image)
                image_token_count = int(prepared["image_token_count"])
            prepared_images.append(
                {
                    **item,
                    **prepared,
                }
            )
            image_token_counts.append(image_token_count)
            replacement = self._image_replacement(model_image_token, image_token_count)
            expanded_parts.append(replacement)
            expanded_parts.append(prompt_parts[idx + 1])
        expanded = "".join(expanded_parts)
        image_token_count = sum(image_token_counts)
        return expanded, {
            **multi_modal_data,
            "image": prepared_images,
            "image_token": model_image_token,
            **self._image_token_id_metadata(),
            "image_token_count": image_token_count,
            "image_token_counts": image_token_counts,
        }

    def prepare_request(self, request: RequestState) -> None:
        mm_data = request.multi_modal_data
        if not mm_data:
            return
        try:
            if not self.supports_multimodal:
                raise ValueError("model does not support multimodal inputs")
            images = mm_data.get("image")
            if not isinstance(images, list) or not images:
                raise ValueError("multi_modal_data.image must be a non-empty list")
            pixel_values_rows = []
            image_grid_thw_rows = []
            for item in images:
                if not isinstance(item, dict):
                    raise ValueError("multi_modal_data.image item must be a dict")
                prepared_pixel_values = item.get("prepared_pixel_values")
                image_grid_thw = item.get("image_grid_thw")
                if prepared_pixel_values is not None and image_grid_thw is not None:
                    pixel_values_rows.append(prepared_pixel_values.to(self.device))
                    image_grid_thw_rows.append(image_grid_thw.to(self.device))
                    continue
                image_position_ids = item.get("image_position_ids")
                if prepared_pixel_values is not None and image_position_ids is not None:
                    pixel_values_rows.append(prepared_pixel_values.to(self.device))
                    image_grid_thw_rows.append(image_position_ids.to(self.device))
                    continue
                if item.get("prepared_image") is not None:
                    image = item["prepared_image"]
                elif isinstance(item.get("image"), str):
                    image = self._load_image(item["image"])
                else:
                    raise ValueError(
                        "multi_modal_data.image item must contain string image url"
                    )
                if self._is_qwen2_vl():
                    prepared = self._prepare_qwen2_vl_image(image)
                    pixel_values_rows.append(
                        prepared["prepared_pixel_values"].to(self.device)
                    )
                    image_grid_thw_rows.append(
                        prepared["image_grid_thw"].to(self.device)
                    )
                else:
                    prepared = self._prepare_gemma4_image(image)
                    pixel_values_rows.append(
                        prepared["prepared_pixel_values"].to(self.device)
                    )
                    if prepared.get("image_position_ids") is not None:
                        image_grid_thw_rows.append(
                            prepared["image_position_ids"].to(self.device)
                        )
            multi_modal_inputs = {
                "pixel_values": torch.cat(pixel_values_rows, dim=0),
                "image_token_count": int(mm_data.get("image_token_count", 0) or 0),
                "image_token_id": int(mm_data.get("image_token_id", -1)),
            }
            if image_grid_thw_rows:
                key = "image_grid_thw" if self._is_qwen2_vl() else "image_position_ids"
                multi_modal_inputs[key] = torch.cat(image_grid_thw_rows, dim=0)
            request.multi_modal_inputs = multi_modal_inputs
            self.prepared_requests += 1
            self.prepared_images += len(images)
        except Exception:
            self.prepare_failures += 1
            raise

    def build_prefill_inputs(
        self, req_dicts: list[RequestState]
    ) -> dict[str, torch.Tensor]:
        mm_requests = [req for req in req_dicts if req.multi_modal_inputs]
        if not mm_requests:
            return {}
        pixel_values_rows = []
        image_grid_thw_rows = []
        image_position_id_rows = []
        image_token_counts = []
        image_token_id: int | None = None
        for req in mm_requests:
            mm_inputs = req.multi_modal_inputs or {}
            pixel_values = mm_inputs.get("pixel_values")
            if pixel_values is None:
                continue
            pixel_values_rows.append(pixel_values)
            image_grid_thw = mm_inputs.get("image_grid_thw")
            if image_grid_thw is not None:
                image_grid_thw_rows.append(image_grid_thw)
            image_position_ids = mm_inputs.get("image_position_ids")
            if image_position_ids is not None:
                image_position_id_rows.append(image_position_ids)
            image_token_counts.append(int(mm_inputs.get("image_token_count", 0) or 0))
            current_image_token_id = int(mm_inputs.get("image_token_id", -1))
            if image_token_id is None:
                image_token_id = current_image_token_id
            elif image_token_id != current_image_token_id:
                raise ValueError("mixed image_token_id multimodal batch")
        if not pixel_values_rows:
            return {}
        total_image_token_count = sum(image_token_counts)
        output = {
            "pixel_values": torch.cat(pixel_values_rows, dim=0),
            "image_token_count": total_image_token_count,
            "image_token_counts": image_token_counts,
            "image_token_id": int(image_token_id if image_token_id is not None else -1),
        }
        if image_grid_thw_rows:
            output["image_grid_thw"] = torch.cat(image_grid_thw_rows, dim=0)
        if image_position_id_rows:
            output["image_position_ids"] = torch.cat(image_position_id_rows, dim=0)
        return output

    def get_multimodal_embeddings(
        self,
        mm_inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        if not mm_inputs or not self.supports_multimodal:
            return None
        self.embedding_requests += 1
        embeddings = self.model.get_multimodal_embeddings(**mm_inputs)
        if embeddings is not None:
            self.embeddings_computed += 1
            if embeddings.dim() == 1:
                self.embedding_feature_dim += int(embeddings.shape[0])
            else:
                self.embedding_feature_dim += int(embeddings.shape[-1])
        return embeddings

    def stats(self) -> dict[str, int | float | bool]:
        return {
            "supports_multimodal": bool(self.supports_multimodal),
            "prepared_requests": self.prepared_requests,
            "prepared_images": self.prepared_images,
            "prepare_failures": self.prepare_failures,
            "embedding_requests": self.embedding_requests,
            "embeddings_computed": self.embeddings_computed,
            "embedding_feature_dim_total": self.embedding_feature_dim,
            "avg_embedding_feature_dim": (
                self.embedding_feature_dim / self.embeddings_computed
                if self.embeddings_computed
                else 0.0
            ),
        }

    def reset_stats(self) -> None:
        self.prepared_requests = 0
        self.prepared_images = 0
        self.prepare_failures = 0
        self.embedding_requests = 0
        self.embeddings_computed = 0
        self.embedding_feature_dim = 0

    def _image_token_count(self) -> int:
        vision_config = self._vision_config()
        default_output_length = int(
            getattr(vision_config, "default_output_length", 0) or 0
        )
        if default_output_length > 0:
            return default_output_length
        num_soft_tokens = int(getattr(vision_config, "num_soft_tokens", 0) or 0)
        if num_soft_tokens > 0:
            return num_soft_tokens
        image_size = int(getattr(vision_config, "image_size", 32) or 32)
        patch_size = int(getattr(vision_config, "patch_size", image_size) or image_size)
        if image_size <= 0 or patch_size <= 0 or image_size % patch_size != 0:
            raise ValueError("invalid fixed-grid vision_config for image tokens")
        grid = image_size // patch_size
        return grid * grid

    def _image_token(self) -> str:
        config = getattr(self.model, "config", None)
        token = getattr(config, "image_token", None)
        if token is None and self._is_qwen2_vl():
            return "<|image_pad|>"
        if token is None and self._is_gemma4():
            return "<|image|>"
        return str(token or "<image>")

    def _image_replacement(self, image_token: str, image_token_count: int) -> str:
        if not self._is_gemma4():
            return " ".join([image_token] * image_token_count)
        return f"<|image>{image_token * image_token_count}<image|>"

    def _image_token_id_metadata(self) -> dict[str, int]:
        config = getattr(self.model, "config", None)
        image_token_id = getattr(config, "image_token_id", None)
        if image_token_id is None:
            return {}
        return {"image_token_id": int(image_token_id)}

    def _prompt_image_token(self) -> str:
        if self._is_qwen2_vl():
            return "<image>"
        return self._image_token()

    def _select_prompt_image_token(self, prompt: str, image_count: int) -> str:
        candidates = [self._prompt_image_token()]
        if not self._is_qwen2_vl() and "<|image|>" not in candidates:
            candidates.append("<|image|>")
        if self._is_gemma4() and "<image>" not in candidates:
            candidates.append("<image>")
        matches = [token for token in candidates if prompt.count(token) == image_count]
        if len(matches) == 1:
            return matches[0]
        return candidates[0]

    def _is_gemma4(self) -> bool:
        config = getattr(self.model, "config", None)
        model_type = str(getattr(config, "model_type", "") or "").lower()
        return model_type == "gemma4" or "gemma4" in type(self.model).__name__.lower()

    def _is_qwen2_vl(self) -> bool:
        config = getattr(self.model, "config", None)
        model_type = str(getattr(config, "model_type", "") or "").lower()
        return (
            model_type == "qwen2_vl" or "qwen2vl" in type(self.model).__name__.lower()
        )

    def _prepare_qwen2_vl_image(self, image: Image.Image) -> dict[str, Any]:
        processor = self._get_qwen2_vl_image_processor()
        batch = processor(images=[image], return_tensors="pt")
        pixel_values = batch["pixel_values"]
        image_grid_thw = batch["image_grid_thw"].to(dtype=torch.long)
        return {
            "prepared_pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "image_token_count": self._qwen2_vl_image_token_count(image_grid_thw),
        }

    def _get_qwen2_vl_image_processor(self) -> Any:
        if self._qwen2_vl_image_processor is None:
            from transformers import Qwen2VLImageProcessor

            config = getattr(self.model, "config", None)
            model_path = str(getattr(config, "_name_or_path", "") or "")
            if model_path:
                try:
                    self._qwen2_vl_image_processor = (
                        Qwen2VLImageProcessor.from_pretrained(model_path)
                    )
                except Exception:
                    self._qwen2_vl_image_processor = Qwen2VLImageProcessor()
            else:
                self._qwen2_vl_image_processor = Qwen2VLImageProcessor()
        return self._qwen2_vl_image_processor

    def _qwen2_vl_image_token_count(self, image_grid_thw: torch.Tensor) -> int:
        vision_config = self._vision_config()
        merge = int(getattr(vision_config, "spatial_merge_size", 2) or 2)
        total = 0
        for t, h, w in image_grid_thw.tolist():
            total += int(t) * int(h) * int(w) // (merge * merge)
        return total

    def _prepare_gemma4_image(self, image: Image.Image) -> dict[str, Any]:
        pixel_values, image_position_ids, image_token_count = (
            self._gemma4_image_to_patches(image)
        )
        return {
            "prepared_pixel_values": pixel_values,
            "image_position_ids": image_position_ids,
            "image_token_count": image_token_count,
        }

    def _vision_config(self) -> Any:
        config = getattr(self.model, "config", None)
        vision_config = getattr(config, "vision_config", None)
        if vision_config is None:
            inner = getattr(self.model, "model", None)
            inner_config = getattr(inner, "config", None)
            vision_config = getattr(inner_config, "vision_config", None)
        if vision_config is None:
            raise ValueError("model vision_config is required for multimodal inputs")
        return vision_config

    def _image_to_pixel_values(self, image: Image.Image) -> torch.Tensor:
        pixel_values, _, _ = self._gemma4_image_to_patches(image)
        return pixel_values.to(self.device)

    def _gemma4_image_to_patches(
        self,
        image: Image.Image,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        vision_config = self._vision_config()
        patch_size = int(getattr(vision_config, "patch_size", 16) or 16)
        pool = int(getattr(vision_config, "pooling_kernel_size", 1) or 1)
        max_soft_tokens = self._image_token_count()
        max_patches = max_soft_tokens * pool * pool
        width, height = image.size
        target_height, target_width = self._aspect_ratio_preserving_size(
            height=height,
            width=width,
            patch_size=patch_size,
            max_patches=max_patches,
            pooling_kernel_size=pool,
        )
        rgb = image.convert("RGB").resize(
            (target_width, target_height),
            Image.Resampling.BICUBIC,
        )
        pixels = torch.tensor(list(rgb.getdata()), dtype=torch.float32)
        pixels = pixels.view(target_height, target_width, 3) / 255.0
        patch_rows = target_height // patch_size
        patch_cols = target_width // patch_size
        patches = pixels.view(
            patch_rows,
            patch_size,
            patch_cols,
            patch_size,
            3,
        )
        patches = patches.permute(0, 2, 1, 3, 4).reshape(
            patch_rows * patch_cols,
            patch_size * patch_size * 3,
        )
        grid_x, grid_y = torch.meshgrid(
            torch.arange(patch_cols, dtype=torch.long),
            torch.arange(patch_rows, dtype=torch.long),
            indexing="xy",
        )
        positions = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)
        image_token_count = int(patches.shape[0]) // (pool * pool)
        padding = max_patches - int(patches.shape[0])
        if padding > 0:
            patches = torch.nn.functional.pad(patches, (0, 0, 0, padding))
            positions = torch.nn.functional.pad(positions, (0, 0, 0, padding), value=-1)
        return (
            patches.unsqueeze(0).to(self.device),
            positions.unsqueeze(0).to(self.device),
            image_token_count,
        )

    def _image_patch_grid(self) -> tuple[int, int]:
        token_count = self._image_token_count()
        rows = int(token_count**0.5)
        while rows > 1 and token_count % rows != 0:
            rows -= 1
        cols = token_count // rows
        pool = int(getattr(self._vision_config(), "pooling_kernel_size", 1) or 1)
        return rows * pool, cols * pool

    @staticmethod
    def _aspect_ratio_preserving_size(
        *,
        height: int,
        width: int,
        patch_size: int,
        max_patches: int,
        pooling_kernel_size: int,
    ) -> tuple[int, int]:
        target_px = max_patches * patch_size * patch_size
        factor = math.sqrt(target_px / float(height * width))
        side_mult = pooling_kernel_size * patch_size
        target_height = int(math.floor((factor * height) / side_mult)) * side_mult
        target_width = int(math.floor((factor * width) / side_mult)) * side_mult
        if target_height == 0 and target_width == 0:
            raise ValueError("image is too small for Gemma4 resize")
        max_side = (
            max_patches // (pooling_kernel_size * pooling_kernel_size)
        ) * side_mult
        if target_height == 0:
            target_height = side_mult
            target_width = min(
                int(math.floor(width / height)) * side_mult,
                max_side,
            )
        if target_width == 0:
            target_width = side_mult
            target_height = min(
                int(math.floor(height / width)) * side_mult,
                max_side,
            )
        return target_height, target_width

    @staticmethod
    def _load_image(image_url: str) -> Image.Image:
        if image_url.startswith("data:"):
            _, encoded = image_url.split(",", 1)
            return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
        parsed = urlparse(image_url)
        if parsed.scheme == "file":
            return Image.open(parsed.path).convert("RGB")
        if parsed.scheme in ("http", "https"):
            with urllib.request.urlopen(image_url) as response:
                return Image.open(io.BytesIO(response.read())).convert("RGB")
        return Image.open(image_url).convert("RGB")
