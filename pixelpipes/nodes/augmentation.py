
from pixelpipes import Macro, Input, types, GraphBuilder, Constant
from pixelpipes.nodes.complex import GetElement
from pixelpipes.nodes.image import GetImageProperties, NormalNoise, ConvertDepth, ImageAdd

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

    def expand(self, inputs, parent: "Reference"):

        image, _ = inputs["source"]
        amount, _ = inputs["amount"]

        with GraphBuilder(prefix=parent) as builder:
            properties = GetImageProperties(source=image)
            width = GetElement(properties, element="width")
            height = GetElement(properties, element="height")

            noise = NormalNoise(width=width, height=height, mean=0, std=amount)

            ConvertDepth(source=ImageAdd(source1=ConvertDepth(source=image, depth="DOUBLE"), source2=noise), depth="BYTE", _name=parent)

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

    def expand(self, inputs, parent: "Reference"):

        image, _ = inputs["source"]
        amount, _ = inputs["amount"]

        with GraphBuilder(prefix=parent) as builder:

            ImageAdd(image, amount, _name=parent)

            return builder.nodes()
