from controlnet_aux import MLSDdetector
from PIL.Image import Image

from .control_interface import ControlInterface


class MlsdControl(ControlInterface):
    def get_control_image(self, image: Image) -> Image:
        mlsd = MLSDdetector.from_pretrained("lllyasviel/ControlNet")
        image = mlsd(image)
        return image
