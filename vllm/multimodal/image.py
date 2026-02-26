# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from PIL import Image

def rescale_image_size(
    image: Image.Image, size_factor: float, transpose: int = -1
) -> Image.Image:
    assert image.mode == "RGBA"
    converted = Image.new("RGB", image.size, background_color)
    converted.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return converted

def convert_image_mode(image: Image.Image, to_mode: str):
    if image.mode == to_mode:
        return image
    elif image.mode == "RGBA" and to_mode == "RGB":
        return rgba_to_rgb(image)
    else:
        return image.convert(to_mode)
