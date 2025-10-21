from annotators.canny_control import CannyControl
from annotators.depth_control import DepthControl
from annotators.lineart_control import LineArtControl
from annotators.mlsd_control import MlsdControl
from annotators.normal_control import NormalControl
from annotators.pose_control import PoseControl
from annotators.shuffle_control import ShuffleControl
from annotators.softedge_control import SoftEdgeControl


class ImageControlFactory:
    def create_control(self, controlnet_type: str):
        if controlnet_type == "Canny":
            return CannyControl()
        elif controlnet_type == "Pose":
            return PoseControl()
        elif controlnet_type == "MLSD":
            return MlsdControl()
        elif controlnet_type == "Depth":
            return DepthControl()
        elif controlnet_type == "LineArt":
            return LineArtControl()
        elif controlnet_type == "Shuffle":
            return ShuffleControl()
        elif controlnet_type == "NormalBAE":
            return NormalControl()
        elif controlnet_type == "SoftEdge":
            return SoftEdgeControl()
        else:
            print("Error: Control type not implemented!")
            raise Exception("Error: Control type not implemented!")
