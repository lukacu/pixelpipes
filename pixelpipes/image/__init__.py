
import os
from typing import Optional

from attributee import String, Boolean

from pixelpipes.list import FileList

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

    def __init__(self, typ: types.Type, field: str, grayscale: Optional[bool] = False):
        super().__init__(typ)
        self._grayscale = grayscale
        self._field = field

    def generate(self, parent, resource):
        return ReadImage(filename=resource.access(self._field, parent), grayscale=self._grayscale)


_EXTENSIONS = [".jpg", ".jpeg", ".png"]

def _recursive_search(path):
    for root, _, files in os.walk(path):
        for basename in files:
            if os.path.splitext(basename)[1].lower() in _EXTENSIONS:
                filename = os.path.join(path, root, basename)
                yield filename   

class ImageDirectory(ResourceListSource):

    path = String()
    grayscale = Boolean(default=False)
    recursive = Boolean(default=False)

    def _load(self):
        if not self.recursive:
            files = [os.path.join(self.path, fi) for fi in os.listdir(self.path) if os.path.splitext(fi)[1].lower() in _EXTENSIONS]
        else:
            files = list(_recursive_search(self.path))
        
        return {
            "lists": {
                "file": (FileList, files)}, 
                "size": len(files)
            }

    def fields(self):
        return dict(image=LoadImage(types.Image(channels=1 if self.grayscale else 3, depth=8), field="file", grayscale=self.grayscale), file=types.String())

from .image import *
from .augmentation import *