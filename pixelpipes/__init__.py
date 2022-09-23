from __future__ import absolute_import
from typing import Iterable, List, Mapping, Optional

import os

__version__ = "0.0.7"

import os
import logging
from collections import namedtuple

_logger = logging.getLogger(__name__)

modules_path = os.environ.get("PIXELPIPES_MODULES_PATH", "").split(os.pathsep)
modules_path.insert(0, os.path.dirname(__file__))
os.environ["PIXELPIPES_MODULES_PATH"] = os.pathsep.join(modules_path)

def load_module(name) -> bool:
    from . import pypixelpipes
    _logger.debug("Loading module %s" % name)
    return pypixelpipes.load(name)

def include_dirs() -> List[str]:
    """Returns a list of directories with C++ header files for pixelpipes core library. Useful when building pixelpipes modules.

    Returns:
        List[str]: List of directories
    """
    include_dir = os.path.join(os.path.dirname(__file__), "include")
    if os.path.isdir(include_dir):
        return [include_dir]
    else:
        return []

def link_dirs() -> List[str]:
    """Returns a list of directories where a pixelpipe library can be found. Useful when building pixelpipes modules.

    Returns:
        List[str]: List of directories
    """
    return [os.path.join(os.path.dirname(__file__))]

class LazyLoadEnum(Mapping):

    def __init__(self, name):
        self._name = name
        self._data = None

    def _load(self):
        if not self._data:
            from . import pypixelpipes
            self._data = pypixelpipes.enum(self._name)

    def __getitem__(self, key):
        self._load()
        if isinstance(key, int):
            if key in self._data.values():
                return key
        return self._data[key]

    def __len__(self):
        self._load()
        return len(self._data)

    def __iter__(self):
        self._load()
        return iter(self._data)

ContextFields = LazyLoadEnum("context")
ComparisonOperations = LazyLoadEnum("comparison")
LogicalOperations = LazyLoadEnum("logical")
ArithmeticOperations = LazyLoadEnum("arithmetic")

PipelineOperation = namedtuple("PipelineOperation", ["id", "name", "arguments", "inputs"])

class Pipeline(object):
    """Wrapper for the C++ pipeline object, includes metadata
    """

    def __init__(self, data: Iterable[PipelineOperation], optimize=True):

        from . import pypixelpipes
        from pixelpipes.graph import ValidationException

        if isinstance(data, pypixelpipes.Pipeline):
            self._pipeline = data
        else:
            self._pipeline = pypixelpipes.Pipeline()

            indices = {}

            for op in data:
                input_indices = [indices[id] for id in op.inputs]
                try:
                    indices[op.id] = self._pipeline.append(op.name, op.arguments, input_indices)
                except ValueError as ve:
                    raise ValidationException("Error when adding operation %s: %s" % (op.name, str(ve)))
                assert indices[op.id] >= 0
                #self._debug("{} ({}): {} ({})", indices[op.id], op.id,
                #            op.name, ", ".join(["{} ({})".format(i, n) for i, n in zip(input_indices, op.inputs)]))

            self._pipeline.finalize(optimize=optimize)

    def __len__(self):
        return len(self._operations)

    def run(self, index):
        return self._pipeline.run(index)

    @property
    def outputs(self):
        return self._pipeline.labels()

    def stats(self):
        # TODO: remove this
        stats = self._pipeline.operation_time()
        for k, v in zip(self._operations, stats):
            print("%s: %.3f ms" % (k, v))
        
    def __iter__(self):
        index = 1
        while True:
            yield self._pipeline.run(index)
            index += 1

def write_pipeline(filename: str, pipeline: Pipeline, compress: Optional[bool] = True):
        from . import pypixelpipes
        pypixelpipes.write_pipeline(pipeline._pipeline, filename, compress)

def read_pipeline(filename: str):
    from . import pypixelpipes
    pipeline = pypixelpipes.read_pipeline(filename)
    return Pipeline(pipeline)

def visualize_pipeline(pipeline: Pipeline):
    from . import pypixelpipes
    try:
        from graphviz import Source
    except ImportError:
        raise ImportError("Install graphviz to visualize pipeline")
    graph = Source(pypixelpipes.visualize_pipeline(pipeline._pipeline))
    graph.view()