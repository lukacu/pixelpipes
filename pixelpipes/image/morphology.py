
from .. import types
from ..graph import Operation, Input

class Moments(Operation):
    """Calculates (first five) image moments."""

    source = Input(types.Image(), description="Input image")

    def operation(self):
        return "opencv:moments",

class ConnectedComponents(Operation):
    """Finds connected components in a binary image."""

    source = Input(types.Image(), description="Input image")

    def operation(self):
        return "opencv:connected_components",
    
class DistanceTransform(Operation):
    """Calculates distance to nearest zero pixel in binary image."""

    source = Input(types.Image(), description="Input image")

    def operation(self):
        return "opencv:distance_transform",