import os
import typing
import pickle

import numpy as np

from pixelpipes import Pipeline
from pixelpipes.graph import Node, NodeException, Output, Reference

class Counter(object):
    """Object based counter, each time it is called it returns a value greater by 1"""

    def __init__(self):
        self._i = 0

    def __call__(self) -> int:
        self._i += 1
        return self._i

class PersistentDict:
    """ A dictionary interface to a folder, with memory caching.
    """

    def __init__(self, root: str):
        """Initializes a new persistent dict

        Args:
            root (str): Root folder
        """

        super().__init__()
        self._root = root
        self._memory = {}
        os.makedirs(self._root, exist_ok=True)

    def _filename(self, key: str) -> str:
        """Generates a filename for the given object key.

        Args:
            key (str): Cache key, a single string

        Returns:
            str: Relative path as a string
        """
        return os.path.join(self._root, key)

    def __getitem__(self, key: str) -> typing.Any:
        """Retrieves an image from cache. If it does not exist, a KeyError is raised

        Args:
            key (str): Key of the item

        Raises:
            KeyError: Entry does not exist or cannot be retrieved
            PickleError: Unable to 

        Returns:
            typing.Any: item value
        """

        if key in self._memory:
            return self._memory[key]

        filename = self._filename(key)

        if not os.path.isfile(filename):
            raise KeyError("Unknown key")
        try:
            with open(filename, mode="br") as filehandle:
                data = pickle.load(filehandle)
                self._memory[key] = data
                return data
        except pickle.PickleError as e:
            raise KeyError(e)

    def __setitem__(self, key: str, value: typing.Any) -> None:
        """Sets an item for given key

        Args:
            key (str): Item key
            value (typing.Any): Item value

        """
        filename = self._filename(key)

        self._memory = value

        try:
            with open(filename, "wb") as filehandle:
                pickle.dump(value, filehandle)
        except pickle.PickleError as e:
            print(e)
            pass

    def __delitem__(self, key: str) -> None:
        """Operator for item deletion.

        Args:
            key (str): Key of object to remove
        """
        try:
            filename = self._filename(key)

            try:
                if key in self._memory:
                    del self._memory[key]

                os.unlink(filename)
            except IOError:
                pass
        except KeyError:
            pass

    def __contains__(self, key: str) -> bool:
        """Magic method, does the cache include an item for a given key.

        Args:
            key (str): Item key

        Returns:
            bool: True if object exists for a given key
        """
        filename = self._filename(key)
        return os.path.isfile(filename)

def find_nodes(module=None):
    """Find all nodes in a given module. Returns a list of classes

    Args:
        module (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    from pixelpipes.graph import Node
    import inspect

    if module is None:
        import pixelpipes
        module = pixelpipes

    nodes = []

    for name in dir(module):
        if name.startswith("_"):
            continue
        member = getattr(module, name)
        if inspect.isclass(member) and issubclass(member, Node) and not member.hiddend():
            nodes.append(member)

    return nodes

def _generate_outputs(out):
    from collections.abc import Mapping, Sequence
    if out is None:
        # Perhaps output nodes were defined explicitly, leave this for compiler to find out
        pass
    elif isinstance(out, Mapping):
        for k, v in out.items():
            if not isinstance(v, Node) and not isinstance(v, Reference):
                raise NodeException("Output not a node")
            Output(v, k)
    elif isinstance(out, Sequence):
        for v in out:
            if not isinstance(v, Node) and not isinstance(v, Reference):
                raise NodeException("Output not a node") 
            Output(v, "default")
    elif isinstance(out, Node) or isinstance(out, Reference):
            Output(out, "default")
    else:
        raise NodeException("Output not a node") 

def graph(constructor):

    from pixelpipes.graph import Graph
    def wrapper(*args, **kwargs):
        with Graph() as builder:
            _generate_outputs(constructor(*args, **kwargs))
        return builder.graph()

    return wrapper

def pipeline(variables=None, fixedout=False, debug=False):
    
    from pixelpipes.compiler import Compiler
    from pixelpipes.graph import Graph

    def inner(constructor):
        def wrapper(*args, **kwargs):
            with Graph() as builder:
                _generate_outputs(constructor(*args, **kwargs))
            return Compiler(fixedout=fixedout, debug=debug).build(builder, variables=variables)

        return wrapper
    return inner

def collage(pipeline: Pipeline, index: int, rows: int, columns: int, offset: typing.Optional[int] = 0) -> np.ndarray:
    assert rows > 0 and columns > 0
    assert offset >= 0
    assert index >= 0 and index < len(pipeline.outputs)

    image = None
    for i in range(rows):
        column = []
        for j in range(columns):
            sample = pipeline.run(i * columns + j + 1 + offset)
            column.append(sample[index])
        column = np.concatenate(column, 1)
        image = column if image is None else np.concatenate((image, column), 0)

    return image

def limit(pipeline: Pipeline, field: typing.Union[int, str]):
    """Returns a bounded generator for the pipeline, in every iteration a given field
    value is compared to the current sample number, if the value is reached or supassed
    the generation is interrupted.

    Args:
        pipeline (Pipeline): Original pipeline
        property (typing.Union[int, str]): Either field label or field index

    Yields:
        Tuple: Sample from a pipeline sequence.
    """
    if isinstance(field, str):
        j = pipeline.outputs.index(field)
    else:
        j = field
    assert j >= 0 and j < len(pipeline.outputs)
    i = 1
    for sample in pipeline:
        yield sample
        if sample[j] == i:
            break
        i += 1