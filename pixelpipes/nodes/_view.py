
from attributee import List

from pixelpipes import Node, Macro, Input, GraphBuilder
import pixelpipes.engine as engine
import pixelpipes.types as types

class TranslateView(Node):

    x = Input(types.Float(), default=0)
    y = Input(types.Float(), default=0)

    def _output(self) -> types.Type:
        return types.View()

    def operation(self):
        return engine.TranslateView()

class RotateView(Node):

    angle = Input(types.Float(), default=0)

    def _output(self) -> types.Type:
        return types.View()

    def operation(self):
        return engine.RotateView()

class ScaleView(Node):

    x = Input(types.Float(), default=1)
    y = Input(types.Float(), default=1)

    def _output(self) -> types.Type:
        return types.View()

    def operation(self):
        return engine.ScaleView()

class IdentityView(Node):

    def _output(self) -> types.Type:
        return types.View()

    def operation(self):
        return engine.IdentityView()

class Chain(Node):

    inputs = List(Input(types.View()))

    def _output(self):
        return types.View()     

    def input_values(self):
        return [self.inputs[int(name)] for name, _ in self._gather_inputs()]

    def _gather_inputs(self):
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

    source = Input(types.BoundingBox())

    def _output(self) -> types.Type:
        return types.View()

    def operation(self):
        return engine.CenterView()

class FocusView(Node):

    source = Input(types.BoundingBox())
    scale = Input(types.Float())

    def _output(self) -> types.Type:
        return types.View()

    def operation(self):
        return engine.FocusView()