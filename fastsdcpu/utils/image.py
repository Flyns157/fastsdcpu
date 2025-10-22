from base64 import b64encode, b64decode
from io import BytesIO
from PIL import Image

from ..models.images import ImageFormat


def pil_image_to_base64_str(
    image: Image.Image,
    format: ImageFormat = ImageFormat.JPEG,
) -> str:
    buffer = BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    img_base64 = b64encode(buffer.getvalue()).decode("utf-8")
    return img_base64


def base64_image_to_pil(base64_str) -> Image.Image:
    image_data = b64decode(base64_str)
    image_buffer = BytesIO(image_data)
    image = Image.open(image_buffer)
    return image


def get_blank_image(
    width: int,
    height: int,
) -> Image.Image:
    """
    Create a blank image with the specified width and height.

    Args:
        width (int): The width of the image.
        height (int): The height of the image.

    Returns:
        Image.Image: A blank image with the specified dimensions.
    """
    return Image.new("RGB", (width, height), (0, 0, 0))


def resize_pil_image(
    pil_image: Image.Image,
    image_width,
    image_height,
):
    return pil_image.convert("RGB").resize(
        (
            image_width,
            image_height,
        ),
        Image.Resampling.LANCZOS,
    )
