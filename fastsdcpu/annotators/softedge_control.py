from annotators.control_interface import ControlInterface
from controlnet_aux import PidiNetDetector
from PIL.Image import Image


class SoftEdgeControl(ControlInterface):
    def get_control_image(self, image: Image) -> Image:
        processor = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
        control_image = processor(image)
        return control_image
