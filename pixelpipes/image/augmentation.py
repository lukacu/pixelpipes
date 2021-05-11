
from attributee.primitives import Integer
from pixelpipes.image.geometry import ImageRemap, Resize
from ..node import Macro, Input, Reference, types
from ..graph import GraphBuilder
from ..core import Constant
from ..core.numbers import Round

from ..core.complex import GetElement
from . import GetImageProperties, ConvertDepth
from .arithemtic import ImageAdd
from .render import LinearImage, NormalNoise, UniformNoise

class ImageNoise(Macro):
    """Apply gaussian noise to an image

    Inputs:
      - source: Input image
      - amount: Amount of noise

    Category: image, augmentation, noise
    """
    
    source = Input(types.Image(depth=8))
    amount = Input(types.Float())

    def validate(self, **inputs):
        super().validate(**inputs)
        return inputs["source"]

    def expand(self, inputs, parent: Reference):

        with GraphBuilder(prefix=parent) as builder:
            properties = GetImageProperties(source=inputs["source"])
            width = GetElement(properties, element="width")
            height = GetElement(properties, element="height")

            noise = NormalNoise(width=width, height=height, mean=0, std=inputs["amount"])

            ConvertDepth(ConvertDepth(inputs["source"], "Double") + noise, depth="Byte", _name=parent)

            return builder.nodes()


class ImageBrightness(Macro):
    """Change image brightness

    Inputs:
      - source: Input image
      - amount: Amount of noise

    Category: image, augmentation
    """
    
    source = Input(types.Image(depth=8))
    amount = Input(types.Float())

    def validate(self, **inputs):
        super().validate(**inputs)
        return inputs["source"]

    def expand(self, inputs, parent: Reference):

        with GraphBuilder(prefix=parent) as builder:
            ImageAdd(inputs["source"], inputs["amount"], _name=parent)

            return builder.nodes()


class ImagePiecewiseAffine(Macro):
    """Piecewise affine transformation of image. This augmentation creates a grid of random perturbations and
    interpolates this transformation over the entire image.


    Inputs:
      - source: Input image
      - amount: Amount of petrubations

    Arguments:
      - subdivision: Number of points

    Category: image, augmentation
    """
    
    source = Input(types.Image())
    amount = Input(types.Float())
    subdivision = Integer(val_min=2, default=4)

    def validate(self, **inputs):
        super().validate(**inputs)
        return inputs["source"]

    def expand(self, inputs, parent: Reference):

        with GraphBuilder(prefix=parent) as builder:
            properties = GetImageProperties(inputs["source"])
            width = GetElement(properties, element="width")
            height = GetElement(properties, element="height")

            x = ConvertDepth(Resize(UniformNoise(self.subdivision, self.subdivision, -inputs["amount"], inputs["amount"]), width, height, interpolation="Linear") + LinearImage(width, height, 0, width, flip=False), "Float")
            y = ConvertDepth(Resize(UniformNoise(self.subdivision, self.subdivision, -inputs["amount"], inputs["amount"]), width, height, interpolation="Linear") + LinearImage(width, height, 0, height, flip=True), "Float")

            ImageRemap(inputs["source"], x, y, interpolation="Linear", border="Reflect", _name=parent)

            return builder.nodes()