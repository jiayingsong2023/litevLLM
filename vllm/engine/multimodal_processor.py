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
                if not isinstance(item, dict) or not isinstance(item.get("image"), str):
                    raise ValueError(
                        "multi_modal_data.image item must contain string image url"
                    )
                image = self._load_image(item["image"])
                pixel_values_rows.append(self._image_to_pixel_values(image))
            request.multi_modal_inputs = {
                "pixel_values": torch.cat(pixel_values_rows, dim=0),
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
        return {"pixel_values": pixel_values}

    def get_multimodal_embeddings(
        self,
        mm_inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        if not mm_inputs or not self.supports_multimodal:
            return None
        self.embedding_requests += 1
        embeddings = self.model.get_multimodal_embeddings(**mm_inputs)
        if embeddings is not None:
            embeddings = self._aggregate_request_embeddings(embeddings)
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

    def _image_to_pixel_values(self, image: Image.Image) -> torch.Tensor:
        rgb = image.convert("RGB").resize((32, 32))
        flat = torch.tensor(list(rgb.getdata()), dtype=torch.float32)
        flat = flat.view(-1) / 255.0
        if flat.numel() < 1024:
            flat = torch.nn.functional.pad(flat, (0, 1024 - flat.numel()))
        else:
            flat = flat[:1024]
        return flat.unsqueeze(0).to(self.device)

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
