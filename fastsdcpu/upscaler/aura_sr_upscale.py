from PIL import Image

from ..upscaler.aura_sr import AuraSR


def upscale_aura_sr(image_path: str):

    aura_sr = AuraSR.from_pretrained("fal/AuraSR-v2", device="cpu")
    image_in = Image.open(image_path)  # .resize((256, 256))
    return aura_sr.upscale_4x(image_in)
