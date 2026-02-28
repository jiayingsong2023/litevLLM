# SPDX-License-Identifier: Apache-2.0
from PIL import Image
from .base import MediaIO

class ImageMediaIO(MediaIO):
    def __init__(self, **kwargs):
        super().__init__()
    def _convert_image_mode(self, image):
        return image.convert("RGB")

class ImageEmbeddingMediaIO(MediaIO):
    def __init__(self, **kwargs):
        super().__init__()
