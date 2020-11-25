
from attributee import String, Float, Integer, Map, List, Boolean, Number, Enumeration

from pixelpipes import Node, Input, wrap_pybind_enum
import pixelpipes.engine as engine
import pixelpipes.types as types

class ViewImage(Node):

    source = Input(types.Image())
    view = Input(types.View())
    width = Input(types.Integer())
    height = Input(types.Integer())
    interpolate = Boolean(default=True)
    border = Enumeration(wrap_pybind_enum(engine.BorderStrategy))

    def operation(self):
        return engine.ViewImage(self.interpolate, self.border)

    def validate(self, **inputs):
        super().validate(**inputs)

        source_type = inputs["source"]
        
        return types.Image(inputs["width"].value, inputs["height"].value, source_type.channels, source_type.depth, source_type.purpose)

class ConvertDepth(Node):

    source = Input(types.Image())
    depth = Enumeration(wrap_pybind_enum(engine.ImageDepth))

    def operation(self):
        return engine.ConvertDepth(self.depth)

    def validate(self, **inputs):
        super().validate(**inputs)

        source_type = inputs["source"]
        
        if self.depth == engine.ImageDepth.BYTE:
            depth = 8
        elif self.depth == engine.ImageDepth.SHORT:
            depth = 16
        elif self.depth == engine.ImageDepth.FLOAT:
            depth = 32
        elif self.depth == engine.ImageDepth.DOUBLE:
            depth = 64

        return types.Image(source_type.width, source_type.height, source_type.channels, depth, source_type.purpose)

class Equals(Node):

    source = Input(types.Image(channels=1))
    value = Input(types.Number())

    def operation(self):
        return engine.Equals()

    def validate(self, **inputs):
        super().validate(**inputs)

        source_type = inputs["source"]
        
        return types.Image(source_type.width, source_type.height, 1, 8)

class MaskBoundingBox(Node):

    source = Input(types.Image(channels=1))

    def operation(self):
        return engine.MaskBoundingBox()

    def _output(self):
        return types.List(types.Float(), 4)

# NEW OPERATIONS

class ImageAdd(Node):

    source = Input(types.Image(channels=1))
    add = Input(types.Image(channels=1))

    def operation(self):
        return engine.ImageAdd()

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(source.width, source.height, 1, 8)

class ImageSubtract(Node):

    source = Input(types.Image(channels=1))
    subtract = Input(types.Image(channels=1))

    def operation(self):
        return engine.ImageSubtract()

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(source.width, source.height, 1, 8)

class ImageMultiply(Node):

    source = Input(types.Image(channels=1))
    multiplier = Input(types.Number())

    def operation(self):
        return engine.ImageMultiply()

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(source.width, source.height, 1, 8)

class GaussianNoise(Node):

    width = Input(types.Number())
    height = Input(types.Number())

    def operation(self):
        return engine.ImageSubtract()

    def validate(self, **inputs):
        super().validate(**inputs)

        width = inputs["width"]
        height = inputs["height"]

        return types.Image(width, height, 1, 8)

class UniformNoise(Node):

    width = Input(types.Number())
    height = Input(types.Number())

    def operation(self):
        return engine.ImageSubtract()

    def validate(self, **inputs):
        super().validate(**inputs)

        width = inputs["width"]
        height = inputs["height"]

        return types.Image(width, height, 1, 8)

class ImageDropout(Node):

    source = Input(types.Image(channels=1))
    probability = Input(types.Number())

    def operation(self):
        return engine.ImageSubtract()

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(source.width, source.height, 1, 8)

class RegionBoundingBox(Node):

    top = Input(types.Number())
    bottom = Input(types.Number())
    left = Input(types.Number())
    right = Input(types.Number())

    def operation(self):
        return engine.ImageSubtract()

    def _output(self):
        return types.List(types.Float(), 4)
