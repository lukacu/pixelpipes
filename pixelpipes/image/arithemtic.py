
from ..graph import BinaryOperation, Input, Node
from typing import Optional

from .. import types

def _infer_type(source1: types.Type, source2: Optional[types.Type] = None, expand: bool = True):

    if source2 is None:
        if not isinstance(source1, types.Image):
            raise types.TypeException("Input should be an image")
        return source1

    if not isinstance(source1, types.Image) and not isinstance(source2, types.Image):
        raise types.TypeException("At least one input should be an image")

    if isinstance(source1, types.Image) and isinstance(source2, types.Image):
        image = source1.common(source2)
        if image.channels is None:
            if source1.channels == 1:
                image = image.duplicate(channels=source2.channels)
            if source2.channels == 1:
                image = image.duplicate(channels=source1.channels)
    else:
        image = source1 if isinstance(source1, types.Image) else source2

    #if image.width is None or image.height is None or image.channels is None:
    #    raise types.TypeException("Incompatible images: {} and {}".format(source1, source2))

    return types.Image(image.width, image.height, image.channels, image.depth, image.purpose)


def _register_image_operation(operation, generator):
    Node.register_operation(operation, generator, _infer_type, types.Image(), types.Image())
    Node.register_operation(operation, generator, _infer_type, types.Image(), types.Number())
    Node.register_operation(operation, generator, _infer_type, types.Number(), types.Image())

class ImageAdd(Node):
    """Image Add

    Adds two images or adds a constant number to image values.

    Inputs:
        - source1: source image
        - source2: image or number

    Category: image, basic
    Tags: image
    """

    source1 = Input(types.Union(types.Image(), types.Number()))
    source2 = Input(types.Union(types.Image(), types.Number()))

    def operation(self):
        return "image:add",

    def validate(self, **inputs):
        super().validate(**inputs)

        source1 = inputs["source1"]
        source2 = inputs["source2"]

        return _infer_type(source1, source2)

_register_image_operation(BinaryOperation.ADD, ImageAdd)

class ImageSubtract(Node):
    """Image subtract

    Subtracts two images with same size and number of channels or an image and a number.

    Inputs:
        - source1: source image
        - source2: image or number

    Category: image, basic
    Tags: image
    """

    source1 = Input(types.Union(types.Image(), types.Number()))
    source2 = Input(types.Union(types.Image(), types.Number()))

    def operation(self):
        return "image:subtract",

    def validate(self, **inputs):
        super().validate(**inputs)

        source1 = inputs["source1"]
        source2 = inputs["source2"]

        return _infer_type(source1, source2)

_register_image_operation(BinaryOperation.SUBTRACT, ImageSubtract)

class ImageMultiply(Node):
    """Image multiply

    Multiplies image with another image or scalar (per-element multiplication).

    Inputs:
        - source1: source image
        - source2: image or number

    Category: image, basic
    Tags: image
    """

    source1 = Input(types.Union(types.Image(), types.Number()))
    source2 = Input(types.Union(types.Image(), types.Number()))

    def operation(self):
        return "image:multiply",

    def validate(self, **inputs):
        super().validate(**inputs)

        source1 = inputs["source1"]
        source2 = inputs["source2"]

        return _infer_type(source1, source2)

_register_image_operation(BinaryOperation.MULIPLY, ImageMultiply)
