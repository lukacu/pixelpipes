
import os

from attributee import String, Boolean

from pixelpipes.list import PrefixStringList

from .. import LazyLoadEnum, load_module

# Load dependendent module to avoid loading it with current module in a wrong namespace (this should be resolved someday)
from .. import geometry 
from .. import types

load_module("image")

ImageDepth = LazyLoadEnum("depth")
InterpolationMode = LazyLoadEnum("interpolation")
BorderStrategy = LazyLoadEnum("border")

from ..resource import ResourceListSource, VirtualField

class LoadImage(VirtualField):

    def __init__(self, typ: types.Type, grayscale: bool):
        super().__init__(typ)
        self._grayscale = grayscale

    def generate(self, parent, resource):
        return ReadImage(filename=resource.access("file", parent), grayscale=self._grayscale)

class ImageDirectory(ResourceListSource):

    EXTENSIONS = [".jpg", ".jpeg", ".png"]

    path = String()
    grayscale = Boolean()

    def _load(self):
        files = [fi for fi in os.listdir(self.path) if os.path.splitext(fi)[1].lower() in ImageDirectory.EXTENSIONS]
        path = self.path if self.path.endswith(os.sep) else (self.path + os.sep)
        return {"lists": {
            "file": (PrefixStringList, files, path)}, "size": len(files)}

    def fields(self):
        return dict(image=LoadImage(types.Image(channels=1 if self.grayscale else 3, depth=8), grayscale=self.grayscale), file=types.String())

from .image import *
from .augmentation import *