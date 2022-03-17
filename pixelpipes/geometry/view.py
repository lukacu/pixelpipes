
from attributee import List

from .types import BoundingBox, View
from .. import types
from ..graph import Node, Macro, Input, Reference
from ..graph import GraphBuilder

class TranslateView(Node):
    """Translate View
    
    Create a translation view.

    Inputs:
        - x: X direction translation
        - y: y direction translation

    Category: View
    """

    x = Input(types.Float(), default=0)
    y = Input(types.Float(), default=0)

    def _output(self) -> types.Type:
        return View()

    def operation(self):
        return "geometry:translate",

class RotateView(Node):
    """Rotate View
    
    Create a rotation view.

    Inputs:
        - angle: Rotation angle

    Category: View
    """

    angle = Input(types.Float(), default=0)

    def _output(self) -> types.Type:
        return View()

    def operation(self):
        return "geometry:rotate",

class ScaleView(Node):
    """Scale View
    
    Create a scale view.

    Inputs:
        - x: X direction scaling
        - y: y direction scaling

    Category: View
    """

    x = Input(types.Float(), default=1)
    y = Input(types.Float(), default=1)

    def _output(self) -> types.Type:
        return View()

    def operation(self):
        return "geometry:scale",

class IdentityView(Node):
    """Identity View
    
    Create an identity view.

    Category: View
    """

    def _output(self) -> types.Type:
        return View()

    def operation(self):
        return "geometry:identity",

class Chain(Node):
    """Chain views

    Multiply a series of views

    Inputs:
        - inputs: A list of views

    Category: view, geometry

    """

    inputs = List(Input(View()))

    def _output(self):
        return View()     

    def input_values(self):
        return [self.inputs[int(name)] for name, _ in self.get_inputs()]

    def get_inputs(self):
        return [(str(k), View()) for k, _ in enumerate(self.inputs)]

    def _merge_config(self, config, update):
        for k, v in update.items():
            i = int(k)
            assert i >= 0 and i < len(config["inputs"])
            config["inputs"][i] = v
        return config

    def operation(self):
        return "geometry:chain",

class AffineView(Macro):
    """Affine View
    
    Create an affine transformation view.

    Inputs:
        - x: X direction scaling
        - y: y direction scaling
        - angle: Rotation angle
        - sx: X skew?
        - sy: Y skew?

    Category: View
    """

    x = Input(types.Float(), default=0)
    y = Input(types.Float(), default=0)
    angle = Input(types.Float(), default=0)
    sx = Input(types.Float(), default=1)
    sy = Input(types.Float(), default=1)

    def _output(self) -> types.Type:
        return View()

    def expand(self, inputs, parent: Reference):
        
        with GraphBuilder(prefix=parent) as builder:

            translate = TranslateView(x=inputs["x"], y=inputs["y"])
            rotate = RotateView(angle=inputs["angle"])
            scale = ScaleView(x=inputs["sx"], y=inputs["sy"])
            
            Chain(inputs=[translate, rotate, scale], _name=parent)

            return builder.nodes()

class CenterView(Node):
    """Center View
    
    Create a view that centers to a bounding box.

    Inputs:
        - source: A bounding box type

    Category: View
    """

    source = Input(BoundingBox())

    def _output(self) -> types.Type:
        return View()

    def operation(self):
        return "geometry:center_view",

class FocusView(Node):
    """Focus View
    
    Create a view that centers to a bounding box and scales so that bounding box maintains relative scale.

    Inputs:
        - source: A bounding box type
        - scale: Scaling factor

    Category: View
    """

    source = Input(BoundingBox())
    scale = Input(types.Float())

    def _output(self) -> types.Type:
        return View()

    def operation(self):
        return "geometry:focus_view",

# TODO: register arithmetic operations