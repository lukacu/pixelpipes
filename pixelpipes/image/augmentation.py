
from attributee.primitives import Integer

from .geometry import ImageRemap, Resize
from ..numbers import Round, Add
from ..graph import Macro, Input, SeedInput, types
from . import GetImageProperties, ConvertDepth
from .render import LinearImage, GaussianNoise, UniformNoise


class ImageNoise(Macro):
    """Apply gaussian noise to an image
    """
    
    source = Input(types.Image(), description="Input image")
    amount = Input(types.Float(), description="Amount of noise")
    seed = SeedInput()

    def expand(self, source, amount, seed):
        properties = GetImageProperties(source)
        noise = GaussianNoise(width=properties["width"], height=properties["height"], mean=0, std=amount, seed=seed)
        return ConvertDepth(Add(ConvertDepth(source, "Float"), noise), depth="Char")


class ImageBrightness(Macro):
    """Change image brightness
    """
    
    source = Input(types.Image(depth=8))
    amount = Input(types.Float())

    def expand(self, source, amount):
        return Add(source, Round(amount), saturate=True)

class ImagePiecewiseAffine(Macro):
    """Piecewise affine transformation of image. This augmentation creates a grid of random perturbations and
    interpolates this transformation over the entire image.
    """
    
    source = Input(types.Image(), description="Input image")
    amount = Input(types.Float(), description="Maximum amount of perturbation in pixels")
    subdivision = Integer(val_min=2, default=4, description="Perturbation lattice subdivision")
    seed = SeedInput()

    def expand(self, source, amount, seed):
        properties = GetImageProperties(source)
        width = properties["width"]
        height = properties["height"]

        x = ConvertDepth(Resize(UniformNoise(self.subdivision, self.subdivision, -amount, amount, seed=seed), width, height, interpolation="Linear") + LinearImage(width, height, 0, width, flip=False), "Float")
        y = ConvertDepth(Resize(UniformNoise(self.subdivision, self.subdivision, -amount, amount, seed=seed + 1), width, height, interpolation="Linear") + LinearImage(width, height, 0, height, flip=True), "Float")
        return ImageRemap(source, x, y, interpolation="Linear", border="Reflect")

