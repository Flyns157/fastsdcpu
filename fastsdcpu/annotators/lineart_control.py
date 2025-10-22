from controlnet_aux import LineartDetector
from PIL.Image import Image

from .control_interface import ControlInterface


class LineArtControl(ControlInterface):
    def get_control_image(self, image: Image) -> Image:
        processor = LineartDetector.from_pretrained("lllyasviel/Annotators")
        control_image = processor(image)
        return control_image
