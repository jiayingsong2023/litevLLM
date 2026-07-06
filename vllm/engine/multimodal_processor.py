# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import base64
import io
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
    ) -> None:
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
        if not isinstance(images, list) or len(images) != 1:
            raise ValueError("Phase 2A supports exactly one image per request")
        item = images[0]
        if not isinstance(item, dict) or not isinstance(item.get("image"), str):
            raise ValueError(
                "multi_modal_data.image item must contain string image url"
            )
        image_token = self._image_token()
        if prompt.count(image_token) != 1:
            raise ValueError(
                f"prompt must contain exactly one {image_token} placeholder"
            )
        image = self._load_image(item["image"])
        image_token_count = self._image_token_count()
        expanded = prompt.replace(
            image_token,
            " ".join([image_token] * image_token_count),
            1,
        )
        return expanded, {
            **multi_modal_data,
            "image": [{**item, "prepared_image": image}],
            "image_token_count": image_token_count,
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
            for item in images:
                if not isinstance(item, dict):
                    raise ValueError("multi_modal_data.image item must be a dict")
                prepared_image = item.get("prepared_image")
                if prepared_image is not None:
                    image = prepared_image
                elif isinstance(item.get("image"), str):
                    image = self._load_image(item["image"])
                else:
                    raise ValueError(
                        "multi_modal_data.image item must contain string image url"
                    )
                pixel_values_rows.append(self._image_to_pixel_values(image))
            request.multi_modal_inputs = {
                "pixel_values": torch.cat(pixel_values_rows, dim=0),
                "image_token_count": int(mm_data.get("image_token_count", 0) or 0),
                "image_token_id": int(mm_data.get("image_token_id", -1)),
            }
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
        if len(req_dicts) != 1 or len(mm_requests) != 1:
            raise ValueError(
                "lite multimodal prefill currently supports "
                "single-request image batches only"
            )
        mm_inputs = mm_requests[0].multi_modal_inputs or {}
        pixel_values = mm_inputs.get("pixel_values")
        if pixel_values is None:
            return {}
        return {
            "pixel_values": pixel_values,
            "image_token_count": int(mm_inputs.get("image_token_count", 0) or 0),
            "image_token_id": int(mm_inputs.get("image_token_id", -1)),
        }

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

    @staticmethod
    def _aggregate_request_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.dim() == 2:
            return embeddings.mean(dim=0, keepdim=True)
        if embeddings.dim() >= 3:
            return embeddings.mean(dim=1)
        return embeddings


    def _image_token_count(self) -> int:
        vision_config = self._vision_config()
        default_output_length = int(
            getattr(vision_config, "default_output_length", 0) or 0
        )
        if default_output_length > 0:
            return default_output_length
        image_size = int(getattr(vision_config, "image_size", 32) or 32)
        patch_size = int(getattr(vision_config, "patch_size", image_size) or image_size)
        if image_size <= 0 or patch_size <= 0 or image_size % patch_size != 0:
            raise ValueError("invalid fixed-grid vision_config for image tokens")
        grid = image_size // patch_size
        return grid * grid

    def _image_token(self) -> str:
        config = getattr(self.model, "config", None)
        token = getattr(config, "image_token", None)
        return str(token or "<image>")

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
        vision_config = self._vision_config()
        patch_size = int(getattr(vision_config, "patch_size", 16) or 16)
        rows, cols = self._image_patch_grid()
        rgb = image.convert("RGB").resize((cols * patch_size, rows * patch_size))
        pixels = torch.tensor(list(rgb.getdata()), dtype=torch.float32)
        pixels = pixels.view(rows * patch_size, cols * patch_size, 3)
        pixels = pixels.permute(2, 0, 1).contiguous() / 255.0
        return pixels.unsqueeze(0).to(self.device)

    def _image_patch_grid(self) -> tuple[int, int]:
        token_count = self._image_token_count()
        rows = int(token_count**0.5)
        while rows > 1 and token_count % rows != 0:
            rows -= 1
        cols = token_count // rows
        return rows, cols

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
