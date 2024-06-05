from __future__ import absolute_import

import os
import logging
from types import MappingProxyType
from typing import Iterable, List, Mapping, Optional, Tuple
from collections import namedtuple

__version__ = "0.1.0"

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

def list_operations() -> List[str]:
    """Returns a list of all available operations.
    
    Returns:
        List[str]: List of operation names.
    """

    from . import pypixelpipes
    return pypixelpipes.operations()

class LazyLoadEnum(Mapping):
    """Special enum class used to load mappings from the core library when they are needed for the first time.
    """

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
        try:
            return self._data[key]
        except KeyError:
            raise KeyError(key + " not available, possible values: " + ", ".join(self._data.keys()))

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
DataType = LazyLoadEnum("datatype")

PipelineOperation = namedtuple("PipelineOperation", ["id", "name", "arguments", "inputs"])

class _PipelineMetadata(object):
    """Wrapper for the C++ pipeline metadata structure. This wrapper should be used instead of interacting with the C++ object directly."""

    def __init__(self, pipeline):
        self._pipeline = pipeline

    def __getitem__(self, key):
        """Accesses the metadata storage.
        
        Args:
            key (str): Key to access
            
        Returns: Value associated with the key
        """
        
        key = str(key)
        return self._pipeline.get(key)

    def __setitem__(self, key, value):
        """Sets the metadata value.
        
        Args:
            key (str): Key to set
            value (str): Value to set
        
        Returns: Value associated with the key
        """
        key = str(key)
        value = str(value)
        return self._pipeline.set(key, value)

class Pipeline(object):
    """Wrapper for the C++ pipeline object, includes additional metadata. This wrapper should be used instead of interacting with the C++ object directly.
    """

    def __init__(self, data: Iterable[PipelineOperation], optimize: bool = None):
        """Creates a new pipeline object from a list of operations.
        
        Args:
            data (Iterable[PipelineOperation]): List of operations
            optimize (bool, optional): Optimize the pipeline. Defaults to True.
        """

        from . import pypixelpipes
        from pixelpipes.graph import ValidationException
        
        if optimize is None:
            import os
            optimize = os.environ.get("PIXELPIPES_PIPELINE_OPTIMIZE", "1") == "1"
        
        try:

            if isinstance(data, pypixelpipes.Pipeline):
                self._pipeline = data
            else:
                self._pipeline = pypixelpipes.Pipeline()

                indices = {}

                for op in data:
                    input_indices = [indices[id] for id in op.inputs]
                    try:
                        indices[op.id] = self._pipeline.append(op.name, tuple(op.arguments), input_indices, {"label": op.id})
                    except ValueError as ve:
                        raise ValidationException("Error when adding operation %s: %s" % (op.name, str(ve)))
                    assert indices[op.id] >= 0

                self._pipeline.finalize(optimize=optimize)

        except pypixelpipes.IllegalStateException as e:
            raise ValidationException(str(e))

    def __len__(self):
        return self._pipeline.size()

    def run(self, index: int) ->Tuple["np.ndarray"]:
        """Executes the pipeline for a given index and resturns result

        Args:
            index (int): Index of sample to generate. Starts with 1.

        Returns:
            Tuple[np.ndarray]: Generated sample, a sequence of NumPy objects.
        """
        from . import pypixelpipes
        try:
            return self._pipeline.run(index)
        except pypixelpipes.IllegalStateException as e:
            raise RuntimeError(str(e)) from e

    @property
    def metadata(self) -> MappingProxyType:
        """Accesses the pipeline metadata storage.

        Returns:
            MappingProxyType: A string to string key-value storage. 
        """
        return _PipelineMetadata(self._pipeline)

    @property
    def outputs(self) -> List[Tuple[str, "pixelpipes.types.Token"]]:
        """Returns description of the pipeline outputs.

        Returns:
            List[Tuple[str, "pixelpipes.types.Token"]]: List of tuples with output name and inferred type.
        """
        from pixelpipes import types
        return [(label, types.Token(shape[0], *shape[1:])) for label, shape in self._pipeline.outputs()]

    def __iter__(self):
        index = 1
        while True:
            yield self._pipeline.run(index)
            index += 1

def write_pipeline(filename: str, pipeline: Pipeline, compress: Optional[bool] = True) -> None:
    """Serializes pipeline to a file with optional compression

    Args:
        filename (str): Filename to use.
        pipeline (Pipeline): Pipeline to serialize.
        compress (Optional[bool], optional): Use GZIP compression or not. Defaults to True.
    """
    from . import pypixelpipes
    pypixelpipes.write_pipeline(pipeline._pipeline, filename, compress)

def read_pipeline(filename: str):
    """Reads pipeline from a file. 
    
    Args:
        filename (str): Filename to read from.
        
    Returns: Pipeline object
    """
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

def evaluate_operation(name: str, inputs, arguments):
    from . import pypixelpipes, types

    def _unwrap(input):
        assert isinstance(input, types.Token)
        if hasattr(input, "_native"):
            return input._native

    def _wrap(output):
        shape = output.shape()
        out = types.Token(shape[0], *shape[1:])
        out._native = output
        return out

    return _wrap(pypixelpipes.evaluate(name, [_unwrap(input) for input in inputs], arguments))
    
