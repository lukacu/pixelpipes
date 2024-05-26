
from attributee import List

from .. import types
from ..graph import Operation, Macro, Input

class TranslateView(Operation):
    """Create a 2D translation matrix."""

    x = Input(types.Float(), default=0, description="X direction translation")
    y = Input(types.Float(), default=0, description="Y direction translation")

    def operation(self):
        return "opencv:translate_view2d",

class RotateView(Operation):
    """Create a 2D rotation matrix."""

    angle = Input(types.Float(), default=0, description="Rotation angle in radians")

    def operation(self):
        return "opencv:rotate_view2d",

class ScaleView(Operation):
    """Create a 2D scale matrix.
    """

    x = Input(types.Float(), default=1, description="X direction scaling")
    y = Input(types.Float(), default=1, description="Y direction scaling")

    def operation(self):
        return "opencv:scale_view2d",

class IdentityView(Operation):
    """ Create a 2D identity matrix."""

    def operation(self):
        return "opencv:identity",

class Chain(Operation):
    """Multiply a series of views"""

    inputs = List(Input(types.View()), description="A list of views to multiply")

    def input_values(self):
        return [self.inputs[int(name)] for name, _ in self.get_inputs()]

    def get_inputs(self):
        return [(str(k), types.View()) for k, _ in enumerate(self.inputs)]

    def _merge_config(self, config, update):
        for k, v in update.items():
            i = int(k)
            assert i >= 0 and i < len(config["inputs"])
            config["inputs"][i] = v
        return config

    def operation(self):
        return "opencv:chain_view2d",

class AffineView(Macro):
    """Create an affine transformation view."""

    x = Input(types.Float(), default=0, description="X translation")
    y = Input(types.Float(), default=0, description="Y translation")
    angle = Input(types.Float(), default=0, description="Rotation angle in radians")
    sx = Input(types.Float(), default=1, description="X scaling")
    sy = Input(types.Float(), default=1, description="Y scaling")

    def expand(self, x, y, angle, sx, sy):
    
        translate = TranslateView(x, y)
        rotate = RotateView(angle)
        scale = ScaleView(x=sx, y=sy)
        
        return Chain(inputs=[translate, rotate, scale])

class CenterView(Operation):
    """Create a view that centers to a bounding box."""

    source = Input(types.Rectangle(), description="A bounding box type")

    def _output(self) -> types.Data:
        return types.View()

class FocusView(Operation):
    """Create a view that centers to a bounding box and scales so that bounding box maintains relative scale."""

    source = Input(types.Rectangle(), description="A bounding box type")
    scale = Input(types.Float(), description="Scaling factor")
