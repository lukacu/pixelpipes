
from pixelpipes.graph import Operation, Input

from attributee import Boolean

from .. import types

from . import ImageDepth

class LoadPNGPaletteIndices(Operation):
    """Decodes buffer as PNG as palette indices (no conversion to RGB)"""

    buffer = Input(types.Buffer(), description="Memory buffer encoded as PNG")

    def operation(self):
        return "load_png_palette",

    def infer(self, buffer):
        # TODO: is this correct
        return types.Image(channels=1, depth="uchar")

class ReadImage(Operation):
    """Read image from file with 8-bit per channel depth. Color or grayscale.
    """

    buffer = Input(types.Buffer(), description="Memory buffer with encoded image")
    grayscale = Boolean(
        default=False, description="Convert to grayscale, otherwise convert to color")

    def operation(self):
        if self.grayscale:
            return "opencv:image_read_grayscale",
        return "opencv:image_read_color",

    def infer(self, buffer):
        return types.Image(channels=1 if self.grayscale else 3)


class ReadImageAny(Operation):
    """Read image from file without conversions
    """

    buffer = Input(types.Buffer(), description="Memory buffer with encoded image")

    def operation(self):
        return "opencv:image_read",

    def infer(self, buffer):
        return types.Image()
