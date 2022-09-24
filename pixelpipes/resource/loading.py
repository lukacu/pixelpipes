
import os
from enum import Enum, auto
from typing import Optional

from attributee import String, Boolean, Enumeration, Callable

from . import ResourceField
from .list import ResourceListSource, FileList
from .. import types
from ..image import ReadImage, ReadImageAny
from ..graph import ReadFile


class ColorConversion(Enum):
    UNCHANGED = auto()
    COLOR = auto()
    GRAYSCALE = auto()

class LoadImage(ResourceField):

    def __init__(self, field: str, loader: Optional[bool] = None):
        super().__init__(types.Image())
        self._loader = loader
        self._field = field

    def access(self, parent):
        if self._loader:
            return self._loader(ReadFile(parent.type[self._field].access(parent)))
        return ReadImage(ReadFile(parent.type[self._field].access(parent)))


_EXTENSIONS = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]


def _recursive_search(path):
    for root, _, files in os.walk(path):
        for basename in files:
            if os.path.splitext(basename)[1].lower() in _EXTENSIONS:
                filename = os.path.join(root, basename)
                yield filename

class ImageDirectory(ResourceListSource):

    path = String(description="Root path to search")
    recursive = Boolean(
        default=False, description="Images are collected in subdirectories as well")
    sorted = Boolean(default=True, description="Sort images by filename")
    conversion = Enumeration(
        ColorConversion, default=ColorConversion.COLOR, description="Color conversion options")
    filter = Callable(
        default=None, description="Filtering function, recieves filename, tells if file should be included")

    def load(self):
        if not self.recursive:
            files = [os.path.join(self.path, fi) for fi in os.listdir(
                self.path) if os.path.splitext(fi)[1].lower() in _EXTENSIONS]
        else:
            files = list(_recursive_search(self.path))

        if self.sorted:
            files = sorted(files)

        files = [os.path.abspath(file) for file in files]
        if self.filter is not None:
            files = [file for file in files if self.filter(file)]

        if self.conversion == ColorConversion.COLOR:
            def loader(x): return ReadImage(x, grayscale=False)
        elif self.conversion == ColorConversion.GRAYSCALE:
            def loader(x): return ReadImage(x, grayscale=True)
        elif self.conversion == ColorConversion.UNCHANGED:
            def loader(x): return ReadImageAny(x)

        image = LoadImage(field="file", loader=loader)

        return {
            "file": FileList(files),
            "image": image
        }
