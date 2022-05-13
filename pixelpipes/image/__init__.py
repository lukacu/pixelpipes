
from .. import LazyLoadEnum, load_module

# Load dependendent module to avoid loading it with current module in a wrong namespace (this should be resolved someday)
from .. import geometry 

load_module("image")

ImageDepth = LazyLoadEnum("depth")
InterpolationMode = LazyLoadEnum("interpolation")
BorderStrategy = LazyLoadEnum("border")

from .image import *
from .loading import *
from .augmentation import *