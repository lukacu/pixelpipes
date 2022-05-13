
import os
from enum import Enum, auto
from typing import Optional

from attributee import String, Boolean, Enumeration

from ..list import FileList
from ..resource import ResourceListSource, VirtualField
from .. import types

class ImageReading(Enum):
    UNCHANGED = auto()
    COLOR = auto()
    GRAYSCALE = auto()

class LoadImage(VirtualField):

    def __init__(self, typ: types.Type, field: str, loader: Optional[bool] = None):
        super().__init__(typ)
        self._loader = loader
        self._field = field

    def generate(self, parent, resource):
        if self._loader:
            return self._loader(resource.access(self._field, parent))
        return ReadImage(resource.access(self._field, parent))


_EXTENSIONS = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]

def _recursive_search(path):
    for root, _, files in os.walk(path):
        for basename in files:
            if os.path.splitext(basename)[1].lower() in _EXTENSIONS:
                filename = os.path.join(root, basename)
                yield filename   

class ImageDirectory(ResourceListSource):

    path = String()
    grayscale = Boolean(default=False)
    recursive = Boolean(default=False)
    sorted = Boolean(default=True)
    reading = Enumeration(ImageReading, default=ImageReading.COLOR)

    def _load(self):
        if not self.recursive:
            files = [os.path.join(self.path, fi) for fi in os.listdir(self.path) if os.path.splitext(fi)[1].lower() in _EXTENSIONS]
        else:
            files = list(_recursive_search(self.path))
        
        if self.sorted:
            files = sorted(files)

        files = [os.path.abspath(file) for file in files]

        return {
            "lists": {
                "file": (FileList, files)}, 
                "size": len(files)
            }

    def fields(self):
        if self.reading == ImageReading.COLOR:
            loader = lambda x: ReadImage(x, grayscale=False)
        elif self.reading == ImageReading.GRAYSCALE:
            loader = lambda x: ReadImage(x, grayscale=True)
        elif self.reading == ImageReading.UNCHANGED:
            loader = lambda x: ReadImageAny(x)


        return dict(image=LoadImage(types.Image(), field="file", loader=loader), file=types.String())

from .image import *
from .augmentation import *