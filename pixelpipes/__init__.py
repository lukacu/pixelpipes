from __future__ import absolute_import

try:
    # Make sure PyTorch is loaded before pixelpipes native code as it loads
    # required dependencies
    import torch
except ImportError:
    pass

import typing
import logging
import threading
import bidict

from attributee import Attributee, Integer, Include, String, List, Object, \
    Map, Attribute, AttributeException, Undefined, is_undefined
from attributee.io import Serializable
from attributee.object import class_fullname
from attributee.privitives import to_number

import pixelpipes.types as types
import pixelpipes.engine as engine

_logger = logging.getLogger(__name__)

def wrap_pybind_enum(bindenum):
    mapping = {}
    for argname in dir(bindenum):
        arg = getattr(bindenum, argname)
        if isinstance(arg, bindenum):
            mapping[arg.name] = arg

    return mapping

_CONTEXT_LOCK = threading.Condition()
_CONTEXT = threading.local()

class Input(Attribute):

    def __init__(self, reftype: types.Type, default=Undefined()):
        super().__init__()
        assert isinstance(reftype, types.Type)
        self._type = reftype
        self._default = default if is_undefined(default) else self.coerce(default, {})

    def coerce(self, value, _):
        assert value is not None

        if isinstance(value, Node):
            with _CONTEXT_LOCK:
                builders = getattr(_CONTEXT, "builders", [])
                if len(builders) > 0:
                    return builders[-1].reference(value)
                else:
                    raise AttributeException("Unable to resolve node's reference")

        if isinstance(value, Reference):
            return value

        ref = Reference.parse(value)
        if ref is not None:
            return ref

        if isinstance(self._type, types.Integer):
            return to_number(value, conversion=int)

        if isinstance(self._type, types.Float):
            return to_number(value, conversion=float)

        if isinstance(self._type, types.Number):
            return to_number(value)

        raise AttributeException("Illegal value")

    def dump(self, value):
        if isinstance(value, Reference):
            return str(value)

        return value

    def reftype(self):
        return self._type

class Reference(object):

    def __init__(self, ref: typing.Union[str, "Reference"]):
        if isinstance(ref, Reference):
            ref = ref.name
        assert str is not None and isinstance(ref, str) and ref != ""
        self._ref = ref

    def __str__(self):
        return "@" + self._ref

    def __repr__(self):
        return "<@" + self._ref + ">"

    @property
    def name(self):
        return self._ref

    @staticmethod
    def parse(value):
        if isinstance(value, str) and value.startswith("@"):
            return Reference(value[1:])
        else:
            return None

    def __add__(self, ref):
        if isinstance(ref, Reference):
            return Reference(self.name + ref.name)
        else:
            return Reference(self.name + str(ref))

class InferredReference(Reference):

    def __init__(self, ref: str, typ: types.Type):
        if isinstance(ref, Reference):
            ref = ref.name
        super().__init__(ref)
        self._typ = typ

    @property
    def type(self):
        return self._typ

class Node(Attributee):

    def __init__(self, _name: str = None, _auto: bool = True, **kwargs):
        """[summary]

        Args:
            _name (str, optional): Internal parameter used for context graph builder. Defaults to None.
            _auto (bool, optional): Should a node automatically be added to a context builder. Defaults to True.
        """

        super().__init__(**kwargs)
        if _auto:
            with _CONTEXT_LOCK:
                builders = getattr(_CONTEXT, "builders", [])
                if len(builders) > 0:
                    builders[-1].add(self, _name)

    def duplicate(self, **inputs):
        config = self.dump()
        for k, v in inputs.items():
            assert k in config
            config[k] = v
        return self.__class__(_auto=False, **config)

    def validate(self, **inputs):
        input_types = self._gather_inputs()
        assert len(inputs) == len(input_types)
        for input_name, input_type in input_types:
            if not input_name in inputs:
                raise ValueError("Input '{}' not available".format(input_name))
            if not input_type.castable(inputs[input_name]):
                raise ValueError("Input '{}' not of correct type for type {}: {} to {}".format(input_name,
                    class_fullname(self), input_type, inputs[input_name]))

        return self._output()

    def _output(self) -> types.Type:
        return types.Any()

    def input_types(self):
        return [i for _, i in self._gather_inputs()]

    def input_values(self):
        return [getattr(self, name) for name, _ in self._gather_inputs()]

    def input_names(self):
        return [name for name, _ in self._gather_inputs()]

    def _gather_inputs(self):
        attributes = getattr(self.__class__, "_declared_attributes", {})

        references = []
        for name, attr in attributes.items():
            if isinstance(attr, Input):
                references.append((name, attr.reftype()))
        return references

    def operation(self):
        raise ValueError("Node not converable to operation")



class Macro(Node):

    def expand(self, inputs, parent: "Reference"):
        raise NotImplementedError()

class Output(Node):

    outputs = List(Input(types.Any()))

    def _output(self) -> types.Type:
        return None

    def _gather_inputs(self):
        return [(str(i), types.Any()) for i, _ in enumerate(self.outputs)]

    def input_values(self):
        return [self.outputs[int(name)] for name, _ in self._gather_inputs()]

    def operation(self):
        return engine.Output()

    def duplicate(self, **inputs):
        config = self.dump()
        for k, v in inputs.items():
            i = int(k)
            assert i >= 0 and i < len(config["outputs"])
            config["outputs"][i] = v
        return self.__class__(**config)

class Copy(Node):

    source = Input(types.Any())

    def validate(self, **inputs):
        super().validate(**inputs)
        return inputs["source"]

    def operation(self):
        return engine.Copy()

class GraphBuilder(object):

    def __init__(self, prefix: typing.Optional[typing.Union[str, Reference]] = ""):
        self._nodes = bidict.bidict()
        self._count = 0
        self._prefix = prefix if isinstance(prefix, str) else prefix.name

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

    def add(self, node, name: typing.Optional[typing.Union[str, Reference]] = None):

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
            raise ValueError("Name already exists: {}".format(name))

        self._nodes[name] = node
        return Reference(name)

    def reference(self, node):
        name = self._nodes.inverse[node]
        return Reference(name)

    def nodes(self):
        return self._nodes

    def build(self):

        nodes = {}

        for k, v in self._nodes.items():
            node = v.dump()
            node["type"] = class_fullname(v)
            nodes[k] = node

        return Graph(nodes=nodes)

    def __contains__(self, node):
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


from pixelpipes.engine import Convert
from pixelpipes.compiler import Compiler
from pixelpipes.nodes import *
from pixelpipes.expression import Expression