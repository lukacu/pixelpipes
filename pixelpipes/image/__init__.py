
from .. import LazyLoadEnum, load_module

load_module("opencv")

ImageDepth = LazyLoadEnum("depth")
InterpolationMode = LazyLoadEnum("interpolation")
BorderStrategy = LazyLoadEnum("border")

from .image import *
from .augmentation import *