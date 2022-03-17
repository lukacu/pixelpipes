
from ..graph import Input, Node, SeedInput
from .. import types
from ..geometry.types import BoundingBox

class ImageBlend(Node):
    """Image blend

    Blends two images with weight defined by alpha.

    Inputs:
        - source1: source image 1
        - source2: source image 2
        - alpha: alpha value between 0 and 1

    Category: image, blend
    Tags: image
    """

    source1 = Input(types.Image())
    source2 = Input(types.Image())
    alpha = Input(types.Float())

    def operation(self):
        return "image:blend",

    def validate(self, **inputs):
        super().validate(**inputs)

        source1 = inputs["source1"]

        return types.Image(source1.width, source1.height, source1.channels, source1.depth)


class ImageDropout(Node):
    """Image dropout

    Sets image pixels to zero with probability p.

    Inputs:
        - source: source image
        - probability: probability between 0 and 1

    Category: image, other
    Tags: image
    """

    source = Input(types.Image())
    probability = Input(types.Float())
    seed = SeedInput()

    def operation(self):
        return "image:dropout",

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(source.width, source.height, source.channels, source.depth)

class ImageCoarseDropout(Node):
    """Image coarse dropout

    Divides an image into patches and cuts them with probability p.

    Inputs:
        - source: source image
        - probability: probability between 0 and 1
        - size_percent: patch size of p percent of image size

    Category: image, other
    Tags: image
    """

    source = Input(types.Image())
    probability = Input(types.Float())
    size_percent = Input(types.Float())
    seed = SeedInput()

    def operation(self):
        return "image:coarse_dropout",

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(source.width, source.height, source.channels, source.depth)



class ImageCut(Node):
    """Image cut

    Cut a patch of an image defined by a bounding box.

    Inputs:
        - source: source image
        - bbox: bounding box

    Category: image, other
    Tags: image
    """

    source = Input(types.Image())
    bbox = Input(BoundingBox())

    def operation(self):
        return "image:cut",

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(source.width, source.height, source.channels, source.depth)

class ImageSolarize(Node):
    """Image solarize

    Invert all values above a threshold in images.

    Inputs:
        - source: source image
        - threshold: threshold value

    Category: image, other
    Tags: image
    """

    source = Input(types.Image(channels=1))
    threshold = Input(types.Float())

    def operation(self):
        return "image:solarize",

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(source.width, source.height, source.channels, source.depth)
