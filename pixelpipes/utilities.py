import os
import typing
import pickle

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

    from pixelpipes import Node
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

def graph(constructor):

    from pixelpipes.graph import GraphBuilder
    def wrapper(*args, **kwargs):
        with GraphBuilder() as builder:
            constructor(*args, **kwargs)
        return builder.graph()

    return wrapper

def pipeline(constructor, variables=None, fixedout=False):
    
    from pixelpipes.graph import GraphBuilder
    def wrapper(*args, **kwargs):
        with GraphBuilder() as builder:
            constructor(*args, **kwargs)
        return builder.pipeline(fixedout=fixedout, variables=variables)

    return wrapper