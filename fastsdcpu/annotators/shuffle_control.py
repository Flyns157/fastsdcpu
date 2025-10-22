from controlnet_aux import ContentShuffleDetector
from PIL.Image import Image

from .control_interface import ControlInterface


class ShuffleControl(ControlInterface):
    def get_control_image(self, image: Image) -> Image:
        shuffle_processor = ContentShuffleDetector()
        image = shuffle_processor(image)
        return image
