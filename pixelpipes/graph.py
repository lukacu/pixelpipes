
import typing
import threading
from attributee import Attributee
from attributee.containers import List, Map
from attributee.io import Serializable
from attributee.object import Object, class_fullname
from attributee.primitives import String

import bidict

from pixelpipes.node import Reference
from pixelpipes.node import Node, NodeException

_CONTEXT_LOCK = threading.Condition()
_CONTEXT = threading.local()


class GraphBuilder(object):

    def __init__(self, prefix: typing.Optional[typing.Union[str, Reference]] = ""):
        self._nodes = bidict.bidict()
        self._count = 0
        self._prefix = prefix if isinstance(prefix, str) else prefix.name

    @staticmethod
    def has_default():
        with _CONTEXT_LOCK:
            builders = getattr(_CONTEXT, "builders", [])
            return len(builders) > 0
            
    @staticmethod
    def default():
        with _CONTEXT_LOCK:
            builders = getattr(_CONTEXT, "builders", [])
            if len(builders) > 0:
                return builders[-1]
            else:
                return None

    @staticmethod
    def add_default(node: Node, name: str) -> bool:
        with _CONTEXT_LOCK:
            builders = getattr(_CONTEXT, "builders", [])
            if len(builders) > 0:
                builders[-1].add(node, name)
                return True
            else:
                return False

    def __enter__(self):
        with _CONTEXT_LOCK:
            if not hasattr(_CONTEXT, "builders"):
                _CONTEXT.builders = [self]
            else:
                _CONTEXT.builders.append(self)
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        with _CONTEXT_LOCK:
            builders = getattr(_CONTEXT, "builders", [])
            if len(builders) == 0 or builders[-1] != self:
                raise RuntimeError("Illegal value")
            _CONTEXT.builders.pop()

    def add(self, node: Node, name: typing.Optional[typing.Union[str, Reference]] = None):

        if name is not None:
            name = name if isinstance(name, str) else name.name

        if node in self._nodes.inverse:
            return Reference(self._nodes.inverse[node])

        if name is None:
            name = "node%d" % self._count
            if self._prefix != "":
                name = "." + name
            self._count += 1

        if name.startswith("."):
            name = self._prefix + name

        if name in self._nodes:
            raise NodeException("Name already exists: {}".format(name), node=name)

        self._nodes[name] = node
        return Reference(name)

    def reference(self, node: Node):
        name = self._nodes.inverse[node]
        return Reference(name)

    def nodes(self):
        return dict(**self._nodes)

    def build(self, groups=None):

        nodes = {}

        for k, v in self._nodes.items():
            node = v.dump()
            node["type"] = class_fullname(v)
            nodes[k] = node

        return Graph(nodes=nodes, groups=groups or list())

    def rename(self, node: Node, newname: str):
        if not node in self._nodes.inverse:
            return

        if newname.startswith("."):
            newname = self._prefix + newname

        if newname in self._nodes:
            raise NodeException("Name already exists: {}".format(newname), node=node)

        oldname = self._nodes.inverse[node]

        del self._nodes[oldname]

        self._nodes[newname] = node
        return Reference(newname)

    def __contains__(self, node: Node):
        return node in self._nodes.inverse

    def __add__(self, element):
        if isinstance(element, Node):
            self.add(element)
        elif isinstance(element, GraphBuilder) and element != self:
            for name, node in element.nodes().items():
                self.add(node, name)
        else:
            raise RuntimeError("Illegal value")
        return self

class Graph(Attributee, Serializable):

    name = String(default="Graph")
    nodes = Map(Object(subclass=Node))
    groups = List(String(), default=[])

    def validate(self):
        from .compiler import infer_type

        type_cache = {}

        for k in self.nodes.keys():
            infer_type(Reference(k), self.nodes, type_cache)

        return type_cache
