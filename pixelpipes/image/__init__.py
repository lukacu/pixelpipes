
import os

from attributee import String

from .. import LazyLoadEnum, load_module

# Load dependendent module to avoid loading it with current module in a wrong namespace (this should be resolved someday)
from .. import geometry 
from .. import types

load_module("image")

ImageDepth = LazyLoadEnum("depth")
InterpolationMode = LazyLoadEnum("interpolation")
BorderStrategy = LazyLoadEnum("border")

from ..resource import ResourceListSource

class ImageDirectory(ResourceListSource):

    EXTENSIONS = [".jpg", ".jpeg", ".png"]

    path = String()

    def _load(self):
        files = [fi for fi in os.listdir(self.path) if os.path.splitext(fi)[1].lower() in ImageDirectory.EXTENSIONS]
        path = self.path if self.path.endswith(os.sep) else (self.path + os.sep)
        return {"lists": {"image": (ImageFileList, files, path)}, "size": len(files)}

    def fields(self):
        return dict(image=types.Image(channels=3, depth=8))

from .image import *
from .augmentation import *