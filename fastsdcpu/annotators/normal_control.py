from controlnet_aux import NormalBaeDetector
from PIL.Image import Image

from .control_interface import ControlInterface


class NormalControl(ControlInterface):
    def get_control_image(self, image: Image) -> Image:
        processor = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
        control_image = processor(image)
        return control_image
