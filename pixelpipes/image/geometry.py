
from ..graph import EnumerationInput, Input, Node, ValidationException
from .. import types

from . import BorderStrategy, InterpolationMode

from ..geometry.types import BoundingBox, View

class Scale(Node):
    """Scale

    Scales an image defined by scale factor.

    Inputs:
        - source: source image
        - scale: scale factor

    Category: image, basic
    Tags: image
    """

    source = Input(types.Image())
    scale = Input(types.Float())
    interpolation = EnumerationInput(InterpolationMode, default="Linear")

    def operation(self):
        return "opencv:rescale",

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        width = None
        height = None

        if inputs["scale"].value is not None and source.width is not None:
            width = int(inputs["scale"].value * source.width)

        if inputs["scale"].value is not None and source.height is not None:
            height = int(inputs["scale"].value * source.height)

        return types.Image(width, height, source.channels, source.depth, purpose=source.purpose)

class Transpose(Node):
    """Transpose

    Transposes image, replacing width for height

    Category: image, basic
    Tags: image
    """

    source = Input(types.Image())

    def operation(self):
        return "opencv:transpose",

    def validate(self, **inputs):
        super().validate(**inputs)
        source = inputs["source"]
        return types.Image(source.height, source.width, source.channels, source.depth)

class Rotate90(Node):
    """Rotate

    Rotate an image 90, -90 or 180 degrees.

    Inputs:
        - source: source image
        - clockwise: 1 is clockwise, -1 is counter clockwise, 0 is vertical flip

    Category: image, basic
    Tags: image
    """

    source = Input(types.Image())
    clockwise = Input(types.Integer())

    def operation(self):
        return "opencv:rotate90",

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]
        clockwise = inputs["clockwise"]

        width = source.width
        height = source.height

        if clockwise == -1 or clockwise == 1:
            width = source.height
            height = source.width

        return types.Image(width, height, source.channels, source.depth)


class Flip(Node):
    """Flip

    Flips a 2D array around vertical, horizontal, or both axes.

    Inputs:
        - source: source image

    Category: image, other
    Tags: image
    """

    source = Input(types.Image())
    horizontal = Input(types.Boolean())
    vertical = Input(types.Boolean())

    def operation(self):
        return "opencv:flip",

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]
        # TODO: incorrect
        return types.Image(source.width, source.height, source.channels, source.depth)


class Resize(Node):
    """Resize

    Resize image to given width and height.

    Inputs:
        - source: source image
        - width: resize width
        - height: resize height

    Category: image, basic
    Tags: image
    """

    source = Input(types.Image())
    width = Input(types.Integer())
    height = Input(types.Integer())

    interpolation = EnumerationInput(InterpolationMode, default="Linear")

    def operation(self):
        return "opencv:resize",

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(inputs["width"].value, inputs["height"].value, source.channels, source.depth, purpose=source.purpose)

class MaskBoundingBox(Node):
    """Mask Bounding Box

    Compute a bounding box of a single-channel image and returns bounding box.

    Inputs:
        - source: source image
        
    Category: image, basic
    Tags: image
    """

    source = Input(types.Image(channels=1))

    def operation(self):
        return "opencv:mask_bounds",

    def _output(self):
        return BoundingBox()

class ImageCrop(Node):
    """Image Crop

    Crops an image

    Inputs:
        - source: source image
        - bbox: bounding box

    Category: image, basic
    Tags: image
    """

    source = Input(types.Image())
    bbox = Input(BoundingBox())

    def operation(self):
        return "opencv:crop",

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(source.width, source.height, source.channels, source.depth)

class ViewImage(Node):
    """Image view

    Apply a view transformation to image

    Inputs:
        - source: source image
        - view: view type
        - width: width
        - height: height 
        - interpolation: interpolation type enumeration
        - border: border type enumeration

    Category: image, geometry
    """

    source = Input(types.Image())
    view = Input(View())
    width = Input(types.Integer())
    height = Input(types.Integer())
    interpolation = EnumerationInput(InterpolationMode, default="Linear")
    border = EnumerationInput(BorderStrategy, default="ConstantLow")

    def operation(self):
        return "opencv:view", 

    def validate(self, **inputs):
        super().validate(**inputs)

        source_type = inputs["source"]
        
        return types.Image(inputs["width"].value, inputs["height"].value, source_type.channels, source_type.depth, source_type.purpose)


class ImageRemap(Node):
    """Image Remap

    Remaps an image

    Inputs:
        - source: source image
        - x: image denoting x lookup coordinates
        - y: image denoting y lookup coordinates

    Category: image, geometry
    Tags: image
    """

    source = Input(types.Image())
    x = Input(types.Image(channels=1, depth=32))
    y = Input(types.Image(channels=1, depth=32))
    interpolation = EnumerationInput(InterpolationMode, default="Linear")
    border = EnumerationInput(BorderStrategy, default="ConstantLow")

    def operation(self):
        return "opencv:remap",

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        dest = inputs["x"].common(inputs["y"])

        # TODO: size inference does not work at the moment
        #if dest.width is None or dest.height is None:
        #    raise ValidationException("X and Y lookup maps mush have same size", self)

        return types.Image(dest.width, dest.height, source.channels, source.depth)
