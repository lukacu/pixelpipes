
from attributee import List

from pixelpipes import Node, Macro, Input, GraphBuilder
import pixelpipes.engine as engine
import pixelpipes.types as types

class TranslateView(Node):

    node_name = "Translate"
    node_description = "Create a translation view"
    node_category = "view"

    x = Input(types.Float(), default=0)
    y = Input(types.Float(), default=0)

    def _output(self) -> types.Type:
        return types.View()

    def operation(self):
        return engine.TranslateView()

class RotateView(Node):

    node_name = "Rotate"
    node_description = "Create a rotation view"
    node_category = "view"

    angle = Input(types.Float(), default=0)

    def _output(self) -> types.Type:
        return types.View()

    def operation(self):
        return engine.RotateView()

class ScaleView(Node):

    node_name = "Scale"
    node_description = "Create a scale view"
    node_category = "view"

    x = Input(types.Float(), default=1)
    y = Input(types.Float(), default=1)

    def _output(self) -> types.Type:
        return types.View()

    def operation(self):
        return engine.ScaleView()

class IdentityView(Node):

    node_name = "Identity"
    node_description = "Create an identity view"
    node_category = "view"

    def _output(self) -> types.Type:
        return types.View()

    def operation(self):
        return engine.IdentityView()

class Chain(Node):
    """Chain views

    Multiply a series of views

    Inputs:
     - inputs: A list of views

    Category: view, geometry

    """

    inputs = List(Input(types.View()))

    def _output(self):
        return types.View()     

    def input_values(self):
        return [self.inputs[int(name)] for name, _ in self.get_inputs()]

    def get_inputs(self):
        return [(str(k), types.View()) for k, _ in enumerate(self.inputs)]


    def duplicate(self, **inputs):
        config = self.dump()
        for k, v in inputs.items():
            i = int(k)
            assert i >= 0 and i < len(config["inputs"])
            config["inputs"][i] = v
        return self.__class__(**config)

    def operation(self):
        return engine.Chain()

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
        return types.View()

    def expand(self, inputs, parent: "Reference"):
        
        with GraphBuilder(prefix=parent) as builder:

            translate = TranslateView(x=inputs["x"][0], y=inputs["y"][0])
            rotate = RotateView(angle=inputs["angle"][0])
            scale = ScaleView(x=inputs["sx"][0], y=inputs["sy"][0])
            
            Chain(inputs=[translate, rotate, scale], _name=parent)

            return builder.nodes()

class CenterView(Node):

    node_name = "Center"
    node_description = "Create a view that centers to a bounding box"
    node_category = "view"

    source = Input(types.BoundingBox())

    def _output(self) -> types.Type:
        return types.View()

    def operation(self):
        return engine.CenterView()

class FocusView(Node):

    node_name = "Focus"
    node_description = "Create a view that centers to a bounding box and scales so that bounding box maintains relative scale"
    node_category = "view"

    source = Input(types.BoundingBox())
    scale = Input(types.Float())

    def _output(self) -> types.Type:
        return types.View()

    def operation(self):
        return engine.FocusView()