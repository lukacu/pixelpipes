
from attributee import List

from .types import BoundingBox, View
from .. import types
from ..node import Node, Macro, Input, Reference
from ..graph import GraphBuilder

class TranslateView(Node):

    node_name = "Translate"
    node_description = "Create a translation view"
    node_category = "view"

    x = Input(types.Float(), default=0)
    y = Input(types.Float(), default=0)

    def _output(self) -> types.Type:
        return View()

    def operation(self):
        return "geometry:translate",

class RotateView(Node):

    node_name = "Rotate"
    node_description = "Create a rotation view"
    node_category = "view"

    angle = Input(types.Float(), default=0)

    def _output(self) -> types.Type:
        return View()

    def operation(self):
        return "geometry:rotate",

class ScaleView(Node):

    node_name = "Scale"
    node_description = "Create a scale view"
    node_category = "view"

    x = Input(types.Float(), default=1)
    y = Input(types.Float(), default=1)

    def _output(self) -> types.Type:
        return View()

    def operation(self):
        return "geometry:scale",

class IdentityView(Node):

    node_name = "Identity"
    node_description = "Create an identity view"
    node_category = "view"

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

    node_name = "Affine"
    node_description = "Create an affine transformation view"
    node_category = "view"

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

    node_name = "Center"
    node_description = "Create a view that centers to a bounding box"
    node_category = "view"

    source = Input(BoundingBox())

    def _output(self) -> types.Type:
        return View()

    def operation(self):
        return "geometry:center_view",

class FocusView(Node):

    node_name = "Focus"
    node_description = "Create a view that centers to a bounding box and scales so that bounding box maintains relative scale"
    node_category = "view"

    source = Input(BoundingBox())
    scale = Input(types.Float())

    def _output(self) -> types.Type:
        return View()

    def operation(self):
        return "geometry:focus_view",

# TODO: register arithmetic operations