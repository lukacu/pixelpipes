
from ..graph import Constant, Input, SeedInput, Operation
from .. import types


class _FloatRangeInput(Input):
    """A two-value floating-point range accepted as a graph input."""

    def __init__(self, default, description=""):
        super().__init__(types.FloatList(2), default=None, description=description)
        self._default = [float(v) for v in default]

    def coerce(self, value, context=None):
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError("A range must contain exactly two values")
            return super().coerce(Constant(value=[float(value[0]), float(value[1])]), context)
        return super().coerce(value, context)


class _IntegerRangeInput(Input):
    """A two-value integer range accepted as a graph input."""

    def __init__(self, default, description=""):
        super().__init__(types.IntegerList(2), default=None, description=description)
        self._default = [int(v) for v in default]

    def coerce(self, value, context=None):
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError("A range must contain exactly two values")
            return super().coerce(Constant(value=[int(value[0]), int(value[1])]), context)
        return super().coerce(value, context)

class ImageBlend(Operation):
    """Image blend

    Blends two images with weight defined by alpha.
    """

    a = Input(types.Image(), description="Image A")
    b = Input(types.Image(), description="Image B")
    alpha = Input(types.Float(), description="Alpha value between 0 and 1")

    def operation(self):
        return "opencv:blend",

class ImageNormalize(Operation):
    """Normalizes values between a range determined by the type of image elements, for integer types this is
    min-max of the type, for float it is 0 and 1.
    """

    source = Input(types.Image(), description="Input image")

    def operation(self):
        return "opencv:normalize",

class ImageDropout(Operation):
    """Sets image pixels to zero with probability p."""

    source = Input(types.Image(), description="Input image")
    probability = Input(types.Float(), description="Dropout probability between 0 and 1")
    seed = SeedInput()

    def operation(self):
        return "opencv:dropout",

class ImageCoarseDropout(Operation):
    """Divides an image into patches and cuts them with probability p.
    """

    source = Input(types.Image(), description="Source image")
    probability = Input(types.Float(), description="Dropput probability between 0 and 1")
    size = Input(types.Float(), description="Patch size of p percent of image size")
    seed = SeedInput()

    def operation(self):
        return "opencv:coarse_dropout",

class ImageCut(Operation):
    """Sets a given rectangular region in an image to zero.
    """

    source = Input(types.Image(), description="Input image")
    region = Input(types.Rectangle(), description="Region rectangle")

    def operation(self):
        return "opencv:cut",

class ImageSolarize(Operation):
    """Invert all values above a threshold in images."""

    source = Input(types.Image(), description="Source image")
    threshold = Input(types.Float(), description="Threshold value")

    def operation(self):
        return "opencv:solarize",

class DerivativeX(Operation):
    """Computes the derivative of an image in the x direction."""

    source = Input(types.Image(), description="Input image")

    def operation(self):
        return "opencv:derivative_x",
    
class DerivativeY(Operation):
    """Computes the derivative of an image in the y direction."""

    source = Input(types.Image(), description="Input image")

    def operation(self):
        return "opencv:derivative_y",
    
class Edges(Operation):
    """Computes the edges of an image."""

    source = Input(types.Image(depth="char"), description="Input image")
    threshold1 = Input(types.Float(), default=0, description="Threshold 1")
    threshold2 = Input(types.Float(), default=100, description="Threshold 2")

    def operation(self):
        return "opencv:edges",
    
class Laplacian(Operation):
    """Computes the Laplacian of an image."""

    source = Input(types.Image(), description="Input image")

    def operation(self):
        return "opencv:laplacian",
    
class DistanceTransform(Operation):
    """Computes the distance transform of an image."""

    source = Input(types.Image(), description="Input image")

    def operation(self):
        return "opencv:distance_transform",


class RandomBrightnessContrast(Operation):
    """Randomly adjust brightness and contrast with one sample per image."""

    source = Input(types.Image(), description="Input image")
    brightness_range = _FloatRangeInput((-0.2, 0.2))
    contrast_range = _FloatRangeInput((-0.2, 0.2))
    brightness_by_max = Input(types.Boolean(), default=False)
    ensure_safe_output = Input(types.Boolean(), default=False)
    seed = SeedInput()

    def operation(self):
        return "opencv:random_brightness_contrast",


class ColorJitter(Operation):
    """Randomly adjust brightness, contrast, saturation and hue in random order."""

    source = Input(types.Image(), description="Input image")
    brightness_range = _FloatRangeInput((0.8, 1.2))
    contrast_range = _FloatRangeInput((0.8, 1.2))
    saturation_range = _FloatRangeInput((0.8, 1.2))
    hue_range = _FloatRangeInput((-0.5, 0.5))
    seed = SeedInput()

    def operation(self):
        return "opencv:color_jitter",


class HueSaturationValue(Operation):
    """Randomly shift the hue, saturation and value channels of an RGB image."""

    source = Input(types.Image(), description="Input image")
    hue_shift_range = _FloatRangeInput((-20, 20))
    sat_shift_range = _FloatRangeInput((-30, 30))
    val_shift_range = _FloatRangeInput((-20, 20))
    seed = SeedInput()

    def operation(self):
        return "opencv:hue_saturation_value",


class RGBShift(Operation):
    """Randomly shift the red, green and blue channels independently."""

    source = Input(types.Image(), description="Input image")
    r_shift_range = _FloatRangeInput((-20, 20))
    g_shift_range = _FloatRangeInput((-20, 20))
    b_shift_range = _FloatRangeInput((-20, 20))
    seed = SeedInput()

    def operation(self):
        return "opencv:rgb_shift",


class RandomGamma(Operation):
    """Apply a random power-law correction to normalized image intensities."""

    source = Input(types.Image(), description="Input image")
    gamma_range = _FloatRangeInput((80, 120))
    seed = SeedInput()

    def operation(self):
        return "opencv:random_gamma",


class CLAHE(Operation):
    """Apply contrast-limited adaptive histogram equalization."""

    source = Input(types.Image(), description="Input image")
    clip_range = _FloatRangeInput((1, 4))
    tile_grid_size = _IntegerRangeInput((8, 8))
    seed = SeedInput()

    def operation(self):
        return "opencv:clahe",


class ChannelDropout(Operation):
    """Replace a random number of complete channels with a fill value."""

    source = Input(types.Image(), description="Input image")
    channel_drop_range = _IntegerRangeInput((1, 1))
    fill = Input(types.Float(), default=0)
    seed = SeedInput()

    def operation(self):
        return "opencv:channel_dropout",


class ChannelShuffle(Operation):
    """Randomly permute image channels."""

    source = Input(types.Image(), description="Input image")
    seed = SeedInput()

    def operation(self):
        return "opencv:channel_shuffle",


class AdditiveNoise(Operation):
    """Add normalized uniform, Gaussian, Laplace or beta noise."""

    source = Input(types.Image(), description="Input image")
    noise_type = Input(types.String(), default="uniform")
    noise_range = _FloatRangeInput((-0.05, 0.05))
    per_channel = Input(types.Boolean(), default=False)
    seed = SeedInput()

    def operation(self):
        return "opencv:additive_noise",


class MultiplicativeNoise(Operation):
    """Multiply pixels by a sampled gain, optionally per pixel or channel."""

    source = Input(types.Image(), description="Input image")
    multiplier = _FloatRangeInput((0.9, 1.1))
    per_channel = Input(types.Boolean(), default=False)
    elementwise = Input(types.Boolean(), default=False)
    seed = SeedInput()

    def operation(self):
        return "opencv:multiplicative_noise",


class ShotNoise(Operation):
    """Apply Poisson shot noise in normalized linear intensity space."""

    source = Input(types.Image(), description="Input image")
    scale_range = _FloatRangeInput((0.1, 0.3))
    seed = SeedInput()

    def operation(self):
        return "opencv:shot_noise",


class SaltAndPepper(Operation):
    """Replace a sampled fraction of pixels with minimum or maximum values."""

    source = Input(types.Image(), description="Input image")
    amount_range = _FloatRangeInput((0.01, 0.06))
    salt_vs_pepper_range = _FloatRangeInput((0.4, 0.6))
    seed = SeedInput()

    def operation(self):
        return "opencv:salt_and_pepper",


# Albumentations-compatible names.
Normalize = ImageNormalize
PixelDropout = ImageDropout
CoarseDropout = ImageCoarseDropout
Erasing = ImageCut
Solarize = ImageSolarize
