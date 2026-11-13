


from ..geometry.view import TranslateView
from ..numbers import Floor, Max, Min, Round, SampleUnform, SquareRoot
from ..graph import EnumerationInput, Input, Macro, Operation, SeedInput
from ..tensor import Float, Integer
from .. import types

from . import BorderStrategy, GetImageProperties, InterpolationMode

class Scale(Operation):
    """Scales an image defined by scale factor.
    """

    source = Input(types.Image(), description="Input image")
    scale = Input(types.Float(), description="Scale factor")
    interpolation = EnumerationInput(InterpolationMode, default="Linear")

    def operation(self):
        return "opencv:rescale",

class Transpose(Operation):
    """Transposes image, switching width for height
    """

    source = Input(types.Image(), description="Input image")

    def operation(self):
        return "opencv:transpose",

class Flip(Operation):
    """Flips image around vertical, horizontal, or both axes.
    """

    source = Input(types.Image(), description="Input image")
    horizontal = Input(types.Boolean(), description="Flip horizontally")
    vertical = Input(types.Boolean(), description="Flip vertically")

    def operation(self):
        return "opencv:flip",


class HorizontalFlip(Macro):
    """Flip an image around its horizontal axis."""

    source = Input(types.Image(), description="Input image")

    def expand(self, source):
        return Flip(source=source, horizontal=True, vertical=False)


class VerticalFlip(Macro):
    """Flip an image around its vertical axis."""

    source = Input(types.Image(), description="Input image")

    def expand(self, source):
        return Flip(source=source, horizontal=False, vertical=True)

class Resize(Operation):
    """ Resize image to given width and height."""

    source = Input(types.Image(), description="Input image")
    width = Input(types.Integer(), description="Desired width")
    height = Input(types.Integer(), description="Desired height")

    interpolation = EnumerationInput(InterpolationMode, default="Linear")

    def operation(self):
        return "opencv:resize",

class MaskBoundingBox(Operation):
    """Compute a bounding box of non-zero pixels in a single-channel image and returns bounding box.
    """

    source = Input(types.Image(), description="Input image")

    def operation(self):
        return "opencv:mask_bounds",

class ImageCrop(Operation):
    """Crops an image to a given rectangle"""

    source = Input(types.Image(), description="Input image")
    x = Input(types.Integer(), description="X coordinate of the top-left corner")
    y = Input(types.Integer(), description="Y coordinate of the top-left corner")
    width = Input(types.Integer(), description="Width of the rectangle")
    height = Input(types.Integer(), description="Height of the rectangle")

    def operation(self):
        return "opencv:crop",

class ImageCropFixed(Operation):
    """Fast crop for rectangles guaranteed to be inside the source image.

    The crop coordinates may be runtime values, while width and height must
    be statically known so downstream batching has a fixed shape.
    """

    source = Input(types.Image(), description="Input image")
    x = Input(types.Integer(), description="X coordinate of the top-left corner")
    y = Input(types.Integer(), description="Y coordinate of the top-left corner")
    width = Input(types.Integer(), description="Width of the rectangle")
    height = Input(types.Integer(), description="Height of the rectangle")

    def operation(self):
        return "opencv:crop_fixed",

class ImageCropSafe(Operation):
    """Crops an image to a given rectangle"""

    source = Input(types.Image(), description="Input image")
    x = Input(types.Integer(), description="X coordinate of the top-left corner")
    y = Input(types.Integer(), description="Y coordinate of the top-left corner")
    width = Input(types.Integer(), description="Width of the rectangle")
    height = Input(types.Integer(), description="Height of the rectangle")
    border = EnumerationInput(BorderStrategy, default="ConstantLow")

    def operation(self):
        return "opencv:crop_safe",

class ViewImage(Operation):
    """ Apply a linear transformation to an image and generate a new image based on it."""

    source = Input(types.Image(), description="Input image")
    view = Input(types.View(), description="Transformation matrix")
    width = Input(types.Integer(), description="Output width")
    height = Input(types.Integer(), description="Output height")
    interpolation = EnumerationInput(InterpolationMode, default="Linear")
    border = EnumerationInput(BorderStrategy, default="ConstantLow")

    def operation(self):
        return "opencv:view", 

class ImageRemap(Operation):
    """Remap image pixels based on given X any Y map using interpoation.
    """

    source = Input(types.Image(), description="Input image")
    x = Input(types.Image(channels=1, depth="float"), description="Matrix denoting X lookup coordinates")
    y = Input(types.Image(channels=1, depth="float"), description="Matrix denoting Y lookup coordinates")
    interpolation = EnumerationInput(InterpolationMode, default="Linear")
    border = EnumerationInput(BorderStrategy, default="ConstantLow")

    def operation(self):
        return "opencv:remap",

class RandomPatchView(Macro):
    """Returns a view that focuses on a random patch in an image"""

    source = Input(types.Image(), description="Input image")
    width = Input(types.Integer(), description="Width of a patch")
    height = Input(types.Integer(), description="Height of a patch")
    padding = Input(types.Integer(), default=0, description="Padding for sampling")
    seed = SeedInput()

    def expand(self, source, width, height, padding, seed):

        properties = GetImageProperties(source)
        image_width = properties["width"]
        image_height = properties["height"]

        x = SampleUnform(- padding, image_width - width + padding, seed=seed)
        y = SampleUnform(- padding, image_height - height + padding, seed=seed * 2)

        return TranslateView(x=-x, y=-y)


class CenterCrop(Macro):
    """Crop the centered region of an image to a fixed size.

    If the requested crop is larger than the source, the missing area is
    filled using ``border``. The output size is always ``width`` x ``height``.
    """

    source = Input(types.Image(), description="Input image")
    width = Input(types.Integer(), description="Output width")
    height = Input(types.Integer(), description="Output height")
    border = EnumerationInput(BorderStrategy, default="ConstantLow", description="Padding strategy")

    def expand(self, source, width, height, border):
        properties = GetImageProperties(source)
        image_width = properties["width"]
        image_height = properties["height"]

        x = Integer(Floor((Float(image_width) - Float(width)) / 2))
        y = Integer(Floor((Float(image_height) - Float(height)) / 2))
        return ImageCropSafe(source, x=x, y=y, width=width, height=height, border=border)


class RandomCropFromBorders(Macro):
    """Randomly remove strips from the four image borders."""

    source = Input(types.Image(), description="Input image")
    crop_left = Input(types.Float(), default=0.1, description="Maximum left crop fraction")
    crop_right = Input(types.Float(), default=0.1, description="Maximum right crop fraction")
    crop_top = Input(types.Float(), default=0.1, description="Maximum top crop fraction")
    crop_bottom = Input(types.Float(), default=0.1, description="Maximum bottom crop fraction")
    seed = SeedInput()

    def expand(self, source, crop_left, crop_right, crop_top, crop_bottom, seed):
        properties = GetImageProperties(source)
        image_width = properties["width"]
        image_height = properties["height"]

        left = Integer(SampleUnform(0, crop_left * image_width, seed=seed))
        right = Integer(SampleUnform(0, crop_right * image_width, seed=seed + 1))
        top = Integer(SampleUnform(0, crop_top * image_height, seed=seed + 2))
        bottom = Integer(SampleUnform(0, crop_bottom * image_height, seed=seed + 3))

        width = Max(1, image_width - left - right)
        height = Max(1, image_height - top - bottom)
        return ImageCropSafe(source, x=left, y=top, width=width, height=height, border="ConstantLow")


class Pad(Macro):
    """Add symmetric padding around an image."""

    source = Input(types.Image(), description="Input image")
    padding = Input(types.Integer(), default=0, description="Padding on every side")
    border = EnumerationInput(BorderStrategy, default="ConstantLow", description="Padding strategy")

    def expand(self, source, padding, border):
        properties = GetImageProperties(source)
        image_width = properties["width"]
        image_height = properties["height"]
        width = image_width + 2 * padding
        height = image_height + 2 * padding
        return ImageCropSafe(source, x=-padding, y=-padding,
                             width=width, height=height, border=border)


class PadIfNeeded(Macro):
    """Pad an image up to minimum dimensions without cropping larger images."""

    source = Input(types.Image(), description="Input image")
    min_width = Input(types.Integer(), description="Minimum width")
    min_height = Input(types.Integer(), description="Minimum height")
    border = EnumerationInput(BorderStrategy, default="ConstantLow", description="Padding strategy")

    def expand(self, source, min_width, min_height, border):
        properties = GetImageProperties(source)
        image_width = properties["width"]
        image_height = properties["height"]

        width = Max(image_width, min_width)
        height = Max(image_height, min_height)
        left = Integer(Floor((Float(width) - Float(image_width)) / 2))
        top = Integer(Floor((Float(height) - Float(image_height)) / 2))
        return ImageCropSafe(source, x=-left, y=-top,
                             width=width, height=height, border=border)


class LetterBox(Macro):
    """Resize an image to fit a canvas and pad the remaining area."""

    source = Input(types.Image(), description="Input image")
    width = Input(types.Integer(), description="Output width")
    height = Input(types.Integer(), description="Output height")
    interpolation = EnumerationInput(InterpolationMode, default="Linear")
    border = EnumerationInput(BorderStrategy, default="ConstantLow", description="Padding strategy")

    def expand(self, source, width, height, interpolation, border):
        properties = GetImageProperties(source)
        image_width = properties["width"]
        image_height = properties["height"]
        scale = Min(Float(width) / Float(image_width), Float(height) / Float(image_height))
        resized_width = Max(1, Integer(Round(Float(image_width) * scale)))
        resized_height = Max(1, Integer(Round(Float(image_height) * scale)))
        resized = Resize(source, width=resized_width, height=resized_height,
                         interpolation=interpolation)
        x = Integer(Floor((Float(width) - Float(resized_width)) / 2))
        y = Integer(Floor((Float(height) - Float(resized_height)) / 2))
        return ImageCropSafe(resized, x=-x, y=-y,
                             width=width, height=height, border=border)


class LongestMaxSize(Macro):
    """Resize an image so its longest side equals ``max_size``."""

    source = Input(types.Image(), description="Input image")
    max_size = Input(types.Integer(), description="Target longest side")
    interpolation = EnumerationInput(InterpolationMode, default="Linear")

    def expand(self, source, max_size, interpolation):
        properties = GetImageProperties(source)
        image_width = properties["width"]
        image_height = properties["height"]
        longest = Max(image_width, image_height)
        scale = Float(max_size) / Float(longest)
        width = Max(1, Integer(Round(Float(image_width) * scale)))
        height = Max(1, Integer(Round(Float(image_height) * scale)))
        return Resize(source, width=width, height=height, interpolation=interpolation)


class SmallestMaxSize(Macro):
    """Resize an image so its shortest side equals ``max_size``."""

    source = Input(types.Image(), description="Input image")
    max_size = Input(types.Integer(), description="Target shortest side")
    interpolation = EnumerationInput(InterpolationMode, default="Linear")

    def expand(self, source, max_size, interpolation):
        properties = GetImageProperties(source)
        image_width = properties["width"]
        image_height = properties["height"]
        shortest = Min(image_width, image_height)
        scale = Float(max_size) / Float(shortest)
        width = Max(1, Integer(Round(Float(image_width) * scale)))
        height = Max(1, Integer(Round(Float(image_height) * scale)))
        return Resize(source, width=width, height=height, interpolation=interpolation)


class RandomResizedCrop(Macro):
    """Randomly crop by area/aspect-ratio ranges, then resize to a fixed size."""

    source = Input(types.Image(), description="Input image")
    width = Input(types.Integer(), description="Output width")
    height = Input(types.Integer(), description="Output height")
    scale_min = Input(types.Float(), default=0.08, description="Minimum crop area fraction")
    scale_max = Input(types.Float(), default=1.0, description="Maximum crop area fraction")
    ratio_min = Input(types.Float(), default=0.75, description="Minimum crop aspect ratio")
    ratio_max = Input(types.Float(), default=1.3333333333333333, description="Maximum crop aspect ratio")
    interpolation = EnumerationInput(InterpolationMode, default="Linear")
    seed = SeedInput()

    def expand(self, source, width, height, scale_min, scale_max,
               ratio_min, ratio_max, interpolation, seed):
        properties = GetImageProperties(source)
        image_width = properties["width"]
        image_height = properties["height"]
        image_area = Float(image_width) * Float(image_height)

        area = SampleUnform(scale_min, scale_max, seed=seed) * image_area
        ratio = SampleUnform(ratio_min, ratio_max, seed=seed + 1)
        crop_width = Integer(Round(SquareRoot(Max(1.0, area * ratio))))
        crop_height = Integer(Round(SquareRoot(Max(1.0, area / ratio))))
        crop_width = Max(1, Min(image_width, crop_width))
        crop_height = Max(1, Min(image_height, crop_height))

        max_x = Max(0, image_width - crop_width)
        max_y = Max(0, image_height - crop_height)
        x = Integer(Round(SampleUnform(0, max_x, seed=seed + 2)))
        y = Integer(Round(SampleUnform(0, max_y, seed=seed + 3)))
        cropped = ImageCropSafe(source, x=x, y=y, width=crop_width, height=crop_height,
                                border="ConstantLow")
        return Resize(cropped, width=width, height=height, interpolation=interpolation)


class RandomSizedCrop(Macro):
    """Randomly crop a height range at a fixed aspect ratio, then resize."""

    source = Input(types.Image(), description="Input image")
    min_height = Input(types.Integer(), description="Minimum crop height")
    max_height = Input(types.Integer(), description="Maximum crop height")
    width = Input(types.Integer(), description="Output width")
    height = Input(types.Integer(), description="Output height")
    w2h_ratio = Input(types.Float(), default=1.0, description="Crop width/height ratio")
    interpolation = EnumerationInput(InterpolationMode, default="Linear")
    seed = SeedInput()

    def expand(self, source, min_height, max_height, width, height,
               w2h_ratio, interpolation, seed):
        properties = GetImageProperties(source)
        image_width = properties["width"]
        image_height = properties["height"]

        crop_height = Integer(Round(SampleUnform(min_height, max_height, seed=seed)))
        crop_height = Max(1, Min(image_height, crop_height))
        crop_width = Integer(Round(Float(crop_height) * Float(w2h_ratio)))
        crop_width = Max(1, Min(image_width, crop_width))

        max_x = Max(0, image_width - crop_width)
        max_y = Max(0, image_height - crop_height)
        x = Integer(Round(SampleUnform(0, max_x, seed=seed + 1)))
        y = Integer(Round(SampleUnform(0, max_y, seed=seed + 2)))
        cropped = ImageCropSafe(source, x=x, y=y, width=crop_width, height=crop_height,
                                border="ConstantLow")
        return Resize(cropped, width=width, height=height, interpolation=interpolation)

class EnsureSize(Macro):
    """Ensures that the image has at least specific size."""
    
    source = Input(types.Image(), description="Input image")
    width = Input(types.Integer(), description="Minimum width")
    height = Input(types.Integer(), description="Minimum height")
    
    def expand(self, source, width, height):
        from pixelpipes.flow import Conditional
        from pixelpipes.numbers import Max, LogicalOr
        from pixelpipes.tensor import Float

        properties = GetImageProperties(source)
        image_width = properties["width"]
        image_height = properties["height"]
                
        c = LogicalOr((image_width < self.width), (image_height < self.height))
        return Conditional(Scale(source, Max(width / Float(image_width), height / Float(image_height))), source, c)


# Aliases for Albumentations-compatible names.
Crop = ImageCrop
        
