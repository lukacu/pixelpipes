
from numbers import Number

from pixelpipes.graph import GraphBuilder
from pixelpipes.graph import Input, Macro, Node, hidden
import numpy as np

from attributee import Enumeration, Boolean, Attribute, AttributeException, List, String
from attributee.object import Callable

from pixelpipes.list import ListElement
import pixelpipes.types as types

from ..geometry.types import View
from . import InterpolationMode, BorderStrategy, ImageDepth

def ImageProperties(width=None, height=None, channels=None, depth=None):
    return types.Complex({"width": types.Integer(width),
         "height": types.Integer(height),
         "channels": types.Integer(channels),
         "depth": types.Integer(depth)})

def _convert_image(value):
    if isinstance(value, list):
        value = np.array(value, dtype=np.float32)
    elif isinstance(value, Number):
        value = np.array([value])
    elif not isinstance(value, np.ndarray):
        raise AttributeException("Unsupported input")

    width = value.shape[1]
    height = value.shape[0]

    if len(value.shape) > 3:
        raise AttributeException("Unsupported dimensions")

    if len(value.shape) < 3:
        channels = 1
    else:
        channels = value.shape[2]

    if value.dtype == np.uint8:
        depth = 8
    elif value.dtype == np.uint16:
        depth = 16
    elif value.dtype == np.float32:
        depth = 32
    elif value.dtype == np.float64:
        depth = 64
    else:
        raise AttributeException("Unsupported depth")

    return value, types.Image(width, height, channels, depth)

class ImageInput(Attribute):

    def coerce(self, value, _):
        return _convert_image(value)[0]

    def dump(self, value: np.ndarray):
        return value.tolist()

class ConstantImage(Node):
    """Constant in-memory image, preloaded when the graph is compiled

    Provides a way to inject a numpy array object as image into the pipeline.

    Inputs:
     - source: A Numpy type image

    Category: image, input
    """

    source = ImageInput()

    def validate(self, **inputs):
        super().validate(**inputs)
        _, typ = _convert_image(self.source)
        return typ

    def operation(self):
        return "image_constant", self.source

    # Prevent errors when cloning during compilation
    def duplicate(self, **inputs):
        return self

class ConstantImageList(Node):
    """Constant in-memory image list.

    Inputs:
     - source: An image list

    Category: image, input
    """

    source = List(ImageInput())

    def _init(self):
        if len(self.source) == 0:
            self._type = types.List(types.Image(), length=0)
            return

        nchannels = [x.shape[2] if len(x.shape) == 3 else 1 for x in self.source]

        channels = nchannels[0]
        dtype = self.source[0].dtype

        if not all([x == channels for x in nchannels]):
            channels = None
        if not all([x.dtype == dtype for x in self.source]):
            dtype = None

        if dtype == np.uint8:
            depth = 8
        elif dtype == np.int16:
            depth = 16
        elif dtype == np.float32:
            depth = 32
        elif dtype == np.float64:
            depth = 64
        else:
            depth = None

        self._type = types.List(types.Image(channels=channels, depth=depth), length=len(self.source))

    def _output(self):
        return self._type

    def operation(self):
        return "image_list", list(self.source)

    # Prevent errors when cloning during compilation
    def duplicate(self, **inputs):
        return self

class ImageLoader(Node):
    """Constant in-memory image, preloaded when the graph is compiled

    Provides a way to inject a single in-memory image into the pipeline.

    Inputs:
     - loader: A callback that should return the image, it will be called
     internally when data is required. Should be serializable.

    Category: image, input
    """

    loader = Callable()

    def _init(self):
        self._source = None
        
    def _load(self):
        if self._source is not None:
            return self._source
        
        self._source = _convert_image(self.loader())

    def validate(self, **inputs):
        super().validate(**inputs)
        _, typ = self._load()
        return typ

    def operation(self):
        source, _ = self._load()
        return "image_constant", source

    # Prevent errors when cloning during compilation
    def duplicate(self, **inputs):
        return self

@hidden
class _GetImageProperties(Node):
    """Get image properties

    Returns a list of properties of the source image: width, height, channels, depth.
    All four are returned in a integer list.

    Inputs:
     - source: Image for which properties are returned

    Category: image
    """

    source = Input(types.Image())

    def operation(self):
        return "image_properties",

    def validate(self, **inputs):
        super().validate(**inputs)
        i = inputs["source"]
        return types.ConstantList([i.width, i.height, i.channels, i.depth])

class GetImageProperties(Macro):
    """Get image properties

    Returns a structure of properties of the source image: width, height, channels, depth.
    All four elements are integers.

    Inputs:
     - source: Image for which properties are returned

    Category: image
    """

    source = Input(types.Image())

    def validate(self, **inputs):
        super().validate(**inputs)
        return ImageProperties(inputs["source"].width, inputs["source"].height,
            inputs["source"].channels, inputs["source"].depth)

    def expand(self, inputs, parent: "Reference"):

        with GraphBuilder(prefix=parent) as builder:
 
            properties = _GetImageProperties(source=inputs["source"])

            ListElement(parent=properties, index=0, _name=".width")
            ListElement(parent=properties, index=1, _name=".height")
            ListElement(parent=properties, index=2, _name=".channels")
            ListElement(parent=properties, index=3, _name=".depth")

            return builder.nodes()


class ReadImage(Node):
    """Read image from file

    Inputs:
        filename: Input image
        grayscale: Convert to grayscale

    Category: image
    Tags: image, input
    """

    filename = Input(types.String())
    grayscale = Boolean(default=False)

    def operation(self):
        return "image:read", self.grayscale

    def _output(self):
        return types.Image()

class ConvertDepth(Node):
    """Convert depth

    Convert pixel depth of input image

    Inputs:
        source: Input image
        depth: Depth type enumeration

    Category: image
    Tags: image, conversion
    """

    source = Input(types.Image())
    depth = Enumeration(ImageDepth)

    def operation(self):
        return "image:convert", self.depth

    def validate(self, **inputs):
        super().validate(**inputs)

        source_type = inputs["source"]
        
        return types.Image(source_type.width, source_type.height, source_type.channels, self.depth, source_type.purpose)

class Grayscale(Node):
    """Grayscale

    Converts image to grayscale.

    Inputs:
        - source: source image

    Category: image, basic
    Tags: image
    """      

    source = Input(types.Image())

    def operation(self):
        return "image:grayscale",

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(source.width, source.height, 1, source.depth)

class Threshold(Node):
    """Threshold

    Sets pixels with values above threshold to zero

    Inputs:
        - source: source image
        - threshold: threshold value

    Category: image, basic
    Tags: image
    """     

    source = Input(types.Image(channels=1))
    threshold = Input(types.Float())

    def operation(self):
        return "image:threshold",

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(source.width, source.height, 1, source.depth)

class Invert(Node):
    """Invert

    Inverts pixel values

    Inputs:
        - source: source image

    Category: image, basic
    Tags: image
    """   

    source = Input(types.Image())

    def operation(self):
        return "image:invert",

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(source.width, source.height, source.channels, source.depth)
    
class Equals(Node):
    """Equal

    Test if individual pixels match a value, returns binary mask

    Inputs:
        - source: source image
        - value: value to compare

    Category: image, basic
    Tags: image
    """   

    source = Input(types.Image(channels=1))
    #value = Input(types.Float())
    value = Input(types.Integer())

    def operation(self):
        return "image:equals",

    def validate(self, **inputs):
        super().validate(**inputs)

        source_type = inputs["source"]
        
        return types.Image(source_type.width, source_type.height, 1, 8, purpose=types.ImagePurpose.MASK)

class Channel(Node):
    """Channel

    Extracts a single channel from multichannel image.

    Inputs:
        - source: source image
        - index: source image channel index

    Category: image, basic
    Tags: image
    """  

    source = Input(types.Image())
    index = Input(types.Integer())

    def operation(self):
        return "image:channel",

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(source.width, source.height, 1, source.depth)

class Merge(Node):
    """Merge

    Merges three single channel images into three channel image.

    Inputs:
        - source1: source image 1
        - source2: source image 2
        - source2: source image 3

    Category: image, basic
    Tags: image
    """

    # TODO: multi channel

    source1 = Input(types.Image(channels=1))
    source2 = Input(types.Image(channels=1))
    source3 = Input(types.Image(channels=1))

    def operation(self):
        return "image:merge",

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source1"]

        return types.Image(source.width, source.height, 3, source.depth)


class Moments(Node):
    """Moments

    Calculates image moments.

    Inputs:
        - source: source image

    Category: image, basic
    Tags: image
    """

    source = Input(types.Image())
    binary = Boolean(default=True)

    def operation(self):
        return "image:moments", self.binary

    def validate(self, **inputs):
        super().validate(**inputs)

        return types.List(types.Float())
