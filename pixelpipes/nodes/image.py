
import numpy as np

from attributee import Enumeration, Boolean
from attributee.object import Callable

from pixelpipes import Node, Input, wrap_pybind_enum, hidden, GraphBuilder, Macro
from pixelpipes.nodes.list import ListElement
import pixelpipes.engine as engine
import pixelpipes.types as types

def ImageProperties():
    return types.Complex({"width": types.Integer(), "height": types.Integer(), "channels": types.Integer(), "depth": types.Integer()})

@hidden
class ImageLoader(Node):
    """Constant in-memory image, preloaded when the graph is compiled

    Provides a way to inject a single in-memory image into the pipeline.

    Inputs:
     - loader: A callback that should return the image, it will be called
     internally when data is required. Should be serializable.

    Category: image, input
    """

    loader = Callable()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._source = None
        
    def _load(self):
        if self._source is not None:
            return self._source
        
        source = self.loader()

        width = source.shape[1]
        height = source.shape[0]
        if len(source.shape) < 3:
            channels = 1
        else:
            channels = source.shape[2]

        if source.dtype == np.uint8:
            depth = 8
        elif source.dtype == np.int16:
            depth = 16
        elif source.dtype == np.float32:
            depth = 32
        elif source.dtype == np.float64:
            depth = 64
        else:
            raise types.TypeException("Unsupported depth")

        self._source = (source, types.Image(width, height, channels, depth))
        return self._source

    def validate(self, **inputs):
        super().validate(**inputs)
        _, typ = self._load()
        return typ

    def operation(self):
        source, _ = self._load()
        return engine.Constant(source)

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
        return engine.GetImageProperties()

    def validate(self, **inputs):
        super().validate(**inputs)
        return types.List(types.Integer(), 4)

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
        return ImageProperties()

    def expand(self, inputs, parent: "Reference"):

        with GraphBuilder(prefix=parent) as builder:

            source_reference, _ = inputs["source"]
            
            properties = _GetImageProperties(source=source_reference)

            ListElement(parent=properties, index=0, _name=".width")
            ListElement(parent=properties, index=1, _name=".height")
            ListElement(parent=properties, index=2, _name=".channels")
            ListElement(parent=properties, index=3, _name=".depth")

            return builder.nodes()


class ViewImage(Node):
    """Image view

    Apply a view transformation to image

    Inputs:
     - source:
     - view:
     - width:
     - height:
     - interpolation:
     - border:

    Category: image, geometry
    """

    node_name = ""
    node_description = ""
    node_category = "image"

    source = Input(types.Image())
    view = Input(types.View())
    width = Input(types.Integer())
    height = Input(types.Integer())
    interpolation = Enumeration(wrap_pybind_enum(engine.Interpolation), default="LINEAR")
    border = Enumeration(wrap_pybind_enum(engine.BorderStrategy), default="CONSTANT_LOW")

    def operation(self):
        return engine.ViewImage(self.interpolation, self.border)

    def validate(self, **inputs):
        super().validate(**inputs)

        source_type = inputs["source"]
        
        return types.Image(inputs["width"].value, inputs["height"].value, source_type.channels, source_type.depth, source_type.purpose)

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
    depth = Enumeration(wrap_pybind_enum(engine.ImageDepth))

    def operation(self):
        return engine.ConvertDepth(self.depth)

    def validate(self, **inputs):
        super().validate(**inputs)

        source_type = inputs["source"]
        
        depth = 8
        if self.depth == engine.ImageDepth.BYTE:
            depth = 8
        elif self.depth == engine.ImageDepth.SHORT:
            depth = 16
        elif self.depth == engine.ImageDepth.FLOAT:
            depth = 32
        elif self.depth == engine.ImageDepth.DOUBLE:
            depth = 64

        return types.Image(source_type.width, source_type.height, source_type.channels, depth, source_type.purpose)

class Grayscale(Node):
    
    node_name = "Grayscale"
    node_description = "Converts image to grayscale"
    node_category = "image"

    source = Input(types.Image())

    def operation(self):
        return engine.Grayscale()

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(source.width, source.height, 1, source.depth)

class Threshold(Node):
    
    node_name = "Threshold"
    node_description = "Sets pixels with values above threshold to zero"
    node_category = "image"

    source = Input(types.Image(channels=1))
    threshold = Input(types.Float())

    def operation(self):
        return engine.ThresholdImage()

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(source.width, source.height, 1, source.depth)

class Invert(Node):
    
    node_name = "Image invert"
    node_description = "Inverts pixel values"
    node_category = "image"

    source = Input(types.Image())

    def operation(self):
        return engine.Invert()

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(source.width, source.height, source.channels, source.depth)
    
class Equals(Node):

    node_name = "Pixel equal"
    node_description = "Test if individual pixels match a value, returns binary mask"
    node_category = "image"

    source = Input(types.Image(channels=1))
    value = Input(types.Float())

    def operation(self):
        return engine.Equals()

    def validate(self, **inputs):
        super().validate(**inputs)

        source_type = inputs["source"]
        
        return types.Image(source_type.width, source_type.height, 1, 8, purpose=types.ImagePurpose.MASK)

class Channel(Node):
    
    node_name = "Channel"
    node_description = "Extracts a single channel from multichannel image"
    node_category = "image"

    source = Input(types.Image())
    index = Input(types.Integer())

    def operation(self):
        return engine.Channel()

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(source.width, source.height, 1, source.depth)

class Merge(Node):
    
    node_name = "Merge"
    node_description = "Merges 3 single channel images into 3 channel image"
    node_category = "image"

    source_0 = Input(types.Image())
    source_1 = Input(types.Image())
    source_2 = Input(types.Image())

    def operation(self):
        return engine.Merge()

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source_0"]

        return types.Image(source.width, source.height, 3, source.depth)

class Polygon(Node):
    """Draw a polygon to a canvas of a given size

    Inputs:
      - source: list of points
      - width: output width
      - height: height

    Returns:
        [type]: [description]
    """

    source = Input(types.Points())
    width = Input(types.Integer())
    height = Input(types.Integer())

    def operation(self):
        return engine.Polygon()

    def validate(self, **inputs):
        super().validate(**inputs)

        return types.Image(inputs["width"].value, inputs["height"].value, 1, 8, types.ImagePurpose.MASK)

class Moments(Node):
    
    node_name = "Moments"
    node_description = "Calculates image moments."
    node_category = "image"

    source = Input(types.Image())

    def operation(self):
        return engine.Moments()

    def validate(self, **inputs):
        super().validate(**inputs)

        return types.List(types.Float())

class MaskBoundingBox(Node):

    node_name = "Mask bounding box"
    node_description = "Compute a bounding box of a single-channel image and returns bounding box."
    node_category = "image"

    source = Input(types.Image(channels=1))

    def operation(self):
        return engine.MaskBoundingBox()

    def _output(self):
        return types.BoundingBox()

class Resize(Node):

    node_name = "Resize image"
    node_description = "Resize image to given width and height"
    node_category = "image"

    source = Input(types.Image())
    width = Input(types.Integer())
    height = Input(types.Integer())

    interpolation = Enumeration(wrap_pybind_enum(engine.Interpolation), default="LINEAR")

    def operation(self):
        return engine.ImageResize(self.interpolation)

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(inputs["width"].value, inputs["height"].value,
             source.channels, source.depth, purpose=source.purpose)

class Scale(Node):

    node_name = "Scale image"
    node_description = "Scale image by a given factor"
    node_category = "image"

    source = Input(types.Image())
    scale = Input(types.Float())

    interpolation = Enumeration(wrap_pybind_enum(engine.Interpolation), default="LINEAR")

    def operation(self):
        return engine.ImageResize(self.interpolation)

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

# NEW OPERATIONS
# CORE OPERATIONS

class ImageAdd(Node):

    node_name = "Image Add"
    node_description = "Adds two images with same size and number of channels"
    node_category = "image"

    source1 = Input(types.Union(types.Image(), types.Number()))
    source2 = Input(types.Union(types.Image(), types.Number()))

    def operation(self):
        return engine.ImageAdd()

    def validate(self, **inputs):
        super().validate(**inputs)

        source1 = inputs["source1"]
        source2 = inputs["source2"]

        if not isinstance(source1, types.Image) and not isinstance(source2, types.Image):
            raise types.TypeException("At least one input should be an image")

        image = source1 if isinstance(source1, types.Image) else source2

        # TODO: add size verification

        return types.Image(image.width, image.height, image.channels, image.depth)

class ImageSubtract(Node):

    node_name = "Image Subtract"
    node_description = "Subtracts two images with same size and number of channels"
    node_category = "image"

    source_1 = Input(types.Image())
    source_2 = Input(types.Image())

    def operation(self):
        return engine.ImageSubtract()

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source_1"]

        return types.Image(source.width, source.height, source.channels, source.depth)

class ImageMultiply(Node):

    node_name = "Image Multiply"
    node_description = "Multiplies image with a scalar"
    node_category = "image"

    source = Input(types.Image())
    multiplier = Input(types.Float())

    def operation(self):
        return engine.ImageMultiply()

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(source.width, source.height, source.channels, source.depth)

class ImageBlend(Node):

    node_name = "Image Blend"
    node_description = "Blends two images using alpha."
    node_category = "image"

    source_1 = Input(types.Image())
    source_2 = Input(types.Image())
    alpha = Input(types.Number())

    def operation(self):
        return engine.ImageBlend()

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source_1"]

        return types.Image(source.width, source.height, source.channels, source.depth)

# BLURING OPERATIONS

class GaussianFunction(Node):
    """Gaussian function

    Generate a tabulated Gaussian function

    Category: image, function
    """

    size_x = Input(types.Integer())
    size_y = Input(types.Integer())
    mean_x = Input(types.Float())
    mean_y = Input(types.Float())
    sigma_x = Input(types.Float())
    sigma_y = Input(types.Float())
    normalize = Boolean(default=False)
    
    def operation(self):
        return engine.MapFunction(0, self.normalize)

    def validate(self, **inputs):
        super().validate(**inputs)

        return types.Image(inputs["size_x"].value, inputs["size_y"].value, 1, 32, types.ImagePurpose.HEATMAP)

class GaussianBlur(Node):
    
    node_name = "Gaussian Blur"
    node_description = "Blurs an image using a gaussian filter."
    node_category = "image"

    source = Input(types.Image())
    size_x = Input(types.Integer())
    size_y = Input(types.Integer())
    sigma_x = Input(types.Float())
    sigma_y = Input(types.Float())
    
    def operation(self):
        return engine.GaussianBlur()

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(source.width, source.height, source.channels, source.depth)

class MedianBlur(Node):
    """Median Blur

    Blurs an image using a median filter.

    Inputs:
     - source: Source image
     - size: Size of the median window

    Category: image, filters
    """

    source = Input(types.Image())
    size = Input(types.Integer())
    
    def operation(self):
        return engine.MedianBlur()

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(source.width, source.height, source.channels, source.depth)

class AverageBlur(Node):
    """Average blur

    Convolving an image with a normalized box filter.

    Inputs:
        - source: input image
        - size: size of the box filter

    Category: image
    Tags: image, blur
    """

    source = Input(types.Image())
    size = Input(types.Integer())
    
    def operation(self):
        return engine.AverageBlur()

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(source.width, source.height, source.channels, source.depth)

class BilateralFilter(Node):
    
    node_name = "Bilateral Filter"
    node_description = "Applies the bilateral filter to an image."
    node_category = "image"

    source = Input(types.Image())
    d = Input(types.Integer())
    sigma_color = Input(types.Float())
    sigma_space = Input(types.Float())
    
    def operation(self):
        return engine.BilateralFilter()

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(source.width, source.height, source.channels, source.depth)

class ImageFilter(Node):

    node_name = "Image Filter"
    node_description = "Convolves an image with the kernel"
    node_category = "image"

    source = Input(types.Image())
    kernel = Input(types.Image(channels=1))

    def operation(self):
        return engine.ImageFilter()

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(source.width, source.height, source.channels, source.depth)

# NOISE GENERATION

class NormalNoise(Node):

    node_name = "Normal Noise"
    node_description = "Creates a single channel image with values sampled from normal distribution"
    node_category = "image"

    width = Input(types.Integer())
    height = Input(types.Integer())
    mean = Input(types.Float())
    std = Input(types.Float())
    
    def operation(self):
        return engine.NormalNoise()

    def validate(self, **inputs):
        super().validate(**inputs)

        width = inputs["width"]
        height = inputs["height"]

        return types.Image(width, height, 1)

class UniformNoise(Node):

    node_name = "Uniform Noise"
    node_description = "Creates a single channel image with values sampled from uniform distribution"
    node_category = "image"

    width = Input(types.Integer())
    height = Input(types.Integer())
    min = Input(types.Float())
    max = Input(types.Float())

    def operation(self):
        return engine.UniformNoise()

    def validate(self, **inputs):
        super().validate(**inputs)

        width = inputs["width"]
        height = inputs["height"]

        return types.Image(width, height, 1)

# OTHER OPERATIONS

class ImageDropout(Node):

    node_name = "Image Dropout"
    node_description = "Sets image pixels to zero with probability p"
    node_category = "image"

    source = Input(types.Image())
    probability = Input(types.Float())

    def operation(self):
        return engine.ImageDropout()

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(source.width, source.height, source.channels, source.depth)

class ImageCoarseDropout(Node):

    node_name = "Image Coarse Dropout"
    node_description = "Divides an image into patches and cuts them with probability p"
    node_category = "image"

    source = Input(types.Image())
    probability = Input(types.Float())
    size_percent = Input(types.Float())

    def operation(self):
        return engine.ImageCoarseDropout()

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(source.width, source.height, source.channels, source.depth)

class RegionBoundingBox(Node): # NOT WORKING - FIX TODO

    node_name = "Region Bounding Box"
    node_description = "Create a custom size bounding box"
    node_category = "image"

    top = Input(types.Float())
    bottom = Input(types.Float())
    left = Input(types.Float())
    right = Input(types.Float())

    def operation(self):
        return engine.RegionBoundingBox()

    def _output(self):
        return types.BoundingBox()

class ImageCut(Node):

    node_name = "Image Cut"
    node_description = "Cut a patch of an image defined by bounding box"
    node_category = "image"

    source = Input(types.Image())
    bbox = Input(types.BoundingBox())

    def operation(self):
        return engine.ImageCut()

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(source.width, source.height, source.channels, source.depth)

class ImageSolarize(Node):

    node_name = "Image Solarize"
    node_description = "Invert all values above a threshold in images."
    node_category = "image"

    source = Input(types.Image(channels=1))
    threshold = Input(types.Float())

    def operation(self):
        return engine.ImageSolarize()

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(source.width, source.height, source.channels, source.depth)
