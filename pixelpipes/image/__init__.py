
import os

from ..core import load_operation_module, get_enum

# Load dependendent module to avoid loading it with current module in a wrong namespace (this should be resolved someday)
from .. import geometry 

load_operation_module(os.path.join(os.path.dirname(__file__), "libpp_image.so"))

# Import Python extensions
from . import pp_image_py

ImageDepth = get_enum("depth")
InterpolationMode = get_enum("interpolation")
BorderStrategy = get_enum("border")

from .image import *
from .augmentation import *