from __future__ import absolute_import
from importlib.util import module_from_spec
from typing import Iterable, Mapping

import os

__version__ = "0.0.3"

import os
import logging
from collections import namedtuple

import pixelpipes.types as types

_logger = logging.getLogger(__name__)

modules_path = os.environ.get("PIXELPIPES_MODULES_PATH", "").split(os.pathsep)
modules_path.insert(0, os.path.dirname(__file__))
os.environ["PIXELPIPES_MODULES_PATH"] = os.pathsep.join(modules_path)

def load_module(name):
    from . import pypixelpipes
    return pypixelpipes.load(name)

def include_directories():
    # TODO
    root = os.path.dirname(__file__)
    return [os.path.join(root, "core"), os.path.join(root, "geometry"), os.path.join(root, "image")]

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

def write_pipeline(filename: str, operations: Iterable[PipelineOperation]):
        from . import pypixelpipes

        writer = pypixelpipes.PipelineWriter()

        indices = {}
        for op in operations:
            input_indices = [indices[id] for id in op.inputs]
            indices[op.id] = writer.append(op.name, op.arguments, input_indices)
            assert indices[op.id] >= 0

        writer.write(filename)

def read_pipeline(filename: str):
    from . import pypixelpipes

    reader = pypixelpipes.PipelineReader()
    pipeline = reader.read(filename)

    return Pipeline(pipeline)

class Pipeline(object):
    """Wrapper for the C++ pipeline object, includes metadata
    """

    def __init__(self, data: Iterable[PipelineOperation]):

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

            self._pipeline.finalize()

    def __len__(self):
        return len(self._operations)

    def run(self, index):
        return self._pipeline.run(index)

    def outputs(self):
        return self._pipeline.labels()

    def stats(self):
        stats = self._pipeline.operation_time()
        for k, v in zip(self._operations, stats):
            print("%s: %.3f ms" % (k, v))
        