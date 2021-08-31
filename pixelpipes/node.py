
from enum import Enum, auto
import typing
import numbers
import traceback

from attributee import Attributee, Attribute, AttributeException, Undefined, is_undefined

from attributee.object import class_fullname
from attributee.primitives import Enumeration, to_number
from numpy import isin

import pixelpipes.types as types

def wrap_pybind_enum(bindenum):
    mapping = {}
    for argname in dir(bindenum):
        arg = getattr(bindenum, argname)
        if isinstance(arg, bindenum):
            mapping[arg.name] = arg

    return mapping

class UnaryOperation(Enum):

    NEGATE = auto()

class BinaryOperation(Enum):

    ADD = auto()
    SUBTRACT = auto()
    MULIPLY = auto()
    DIVIDE = auto()
    POWER = auto()
    MODULO = auto()

    EQUAL = auto()
    NOT_EQUAL = auto()
    LOWER = auto()
    LOWER_EQUAL = auto()
    GREATER = auto()
    GREATER_EQUAL = auto()

Operation = typing.Union[UnaryOperation, BinaryOperation]

class NodeException(Exception):

    def __init__(self, *args, node: "Node" = None):
        super().__init__(*args)
        self._node = node

    @property
    def node(self):
        return self._node

    def nodetrace(self):
        stack = []
        node = self._node
        while node is not None:
            stack.append(node.frame)
            node = node.origin
        return stack

class ValidationException(NodeException):
    pass


class Input(Attribute):

    def __init__(self, reftype: types.Type, default=Undefined()):
        super().__init__()
        assert isinstance(reftype, types.Type)
        self._type = reftype
        self._default = default if is_undefined(default) else self.coerce(default, {})

    def coerce(self, value, _):
        assert value is not None

        if isinstance(value, Node):
            from pixelpipes.graph import GraphBuilder
            builder = GraphBuilder.default()
            if builder is not None:
                return builder.reference(value)
            else:
                raise AttributeException("Unable to resolve node's reference")

        if isinstance(value, Reference):
            return value

        ref = Reference.parse(value)
        if ref is not None:
            return ref

        if self._type.castable(types.Number()):
            return to_number(value)

        if self._type.castable(types.Float()):
            return to_number(value, conversion=float)

        if self._type.castable(types.Integer()):
            return to_number(value, conversion=int)


        raise AttributeException("Illegal value: {}".format(value))

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
        if not (ref is not None and isinstance(ref, str) and ref != ""):
            raise ValidationException("Reference is undefined")
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

    def __eq__(self, ref):
        if ref is None:
            return False
        if isinstance(ref, Reference):
            return ref.name == self.name
        if isinstance(ref, str):
            return ref == self.name
        return False

class InferredReference(Reference):
    """A node reference with type already inferred. Using during compilation in macro expansion.

    """

    def __init__(self, ref: str, typ: types.Type):
        if isinstance(ref, Reference):
            ref = ref.name
        super().__init__(ref)
        self._typ = typ

    @property
    def type(self):
        return self._typ

    def __getitem__(self, i):
        """Immitate tuple for backwards compatibility"""

        if i == 0:
            return self
        if i == 1:
            return self.type

        if isinstance(self._typ, types.Complex) and isinstance(i, str):
            return self._typ.access(i, self)
        
    def __neg__(self):
        return _UnaryOperationWrapper(self, UnaryOperation.NEGATE)

def hidden(node_class):
    node_class.node_hidden_base = node_class
    return node_class

def _ensure_node(value):
    from .graph import GraphBuilder
    if not GraphBuilder.has_default():
        raise ValueError("Unable to use node operation magic without a builder context")

    if isinstance(value, Reference):
        return value
    if isinstance(value, Node):
        return value
    elif isinstance(value, numbers.Number):
        from .core import Constant
        return Constant(value=value)
    else:
        raise ValueError("Value is not a node or a number")

_operation_registry = {}



@hidden
class Node(Attributee):

    def __init__(self, *args, _name: str = None, _auto: bool = True, _origin: "Node" = None, **kwargs):
        #Node constructor, please do not overload it directly, override _init method to add custom 
        #initialization code.
        #
        #Args:
        #    _name (str, optional): Internal parameter used for context graph builder. Defaults to None.
        #    _auto (bool, optional): Should a node automatically be added to a context builder. Defaults to True.
        
        super().__init__(*args, **kwargs)

        self._cache = {}

        if _auto:
            from pixelpipes.graph import GraphBuilder
            GraphBuilder.add_default(self, _name)

        self._origin = _origin
        self._source = self._nodeframe()
        self._init()

    def _nodeframe(self):
        return traceback.extract_stack()[-2]

    def _init(self):
        pass

    @classmethod
    def name(cls):
        if hasattr(cls, "node_name"):
            return getattr(cls, "node_name")
        else:
            return cls.__name__

    @classmethod
    def description(cls):
        if hasattr(cls, "node_descritption"):
            return getattr(cls, "node_descritption")
        else:
            return ""

    @classmethod
    def category(cls):
        if hasattr(cls, "node_category"):
            return getattr(cls, "node_category")
        else:
            return ""

    @classmethod
    def hidden(cls):
        if hasattr(cls, "node_hidden_base"):
            return getattr(cls, "node_hidden_base") == cls
        else:
            return False

    @property
    def frame(self):
        return self._source

    @property
    def origin(self):
        return self._origin

    def _merge_config(self, config, update):
        for k, v in update.items():
            assert k in config
            config[k] = v
        return config

    def duplicate(self, _name=None, _origin=None, **inputs):
        config = self._merge_config(self.dump(), inputs)
        double = self.__class__(_auto=False, _name=_name, _origin=_origin, **config)
        double._origin = self._origin
        double._source = self._source
        return double

    def validate(self, **inputs):
        input_types = self.get_inputs()
        assert len(inputs) == len(input_types)
        for input_name, input_type in input_types:
            if not input_name in inputs:
                raise ValidationException("Input '{}' not available".format(input_name), node=self)
            if not input_type.castable(inputs[input_name]):
                raise ValidationException("Input '{}' not of correct type for type {}: {} to {}".format(input_name,
                    class_fullname(self), input_type, inputs[input_name]), node=self)

        return self._output()

    def input_types(self):
        return [i for _, i in self.get_inputs()]

    def input_values(self):
        return [getattr(self, name) for name, _ in self.get_inputs()]

    def input_names(self):
        return [name for name, _ in self.get_inputs()]

    def _output(self) -> types.Type:
        return types.Any()

    def get_inputs(self):
        if "inputs" in self._cache:
            return self._cache["inputs"]

        references = []
        for name, attr in self.attributes().items():
            if isinstance(attr, Input):
                references.append((name, attr.reftype()))

        self._cache["inputs"] = references

        return references

    @property
    def origin(self):
        """Only relevant for nodes generated during compilation, makes it easier to 
        track original node from the source graph. Returns None in other cases.
        """
        return self._origin

    def operation(self) -> typing.Tuple:
        raise NodeException("Node not converable to operation", node=self)

    def __hash__(self):
        return id(self)

    def __add__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(self, other, BinaryOperation.ADD)

    def __radd__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(other, self, BinaryOperation.ADD)

    def __sub__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(self, other, BinaryOperation.SUBTRACT)

    def __rsub__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(other, self, BinaryOperation.SUBTRACT)

    def __mul__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(self, other, BinaryOperation.MULIPLY)

    def __rmul__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(other, self, BinaryOperation.MULIPLY)

    def __truediv__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(self, other, BinaryOperation.DIVIDE)

    def __rtruediv__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(other, self, BinaryOperation.DIVIDE)

    def __pow__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(self, other, BinaryOperation.POWER)

    def __rpow__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(other, self, BinaryOperation.POWER)

    def __mod__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(self, other, BinaryOperation.MODULO)

    def __rmod__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(other, self, BinaryOperation.MODULO)

    def __neg__(self):
        return _UnaryOperationWrapper(self, UnaryOperation.NEGATE)

    def __lt__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(self, other, BinaryOperation.LOWER)

    def __le__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(self, other, BinaryOperation.LOWER_EQUAL)

    def __gt__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(self, other, BinaryOperation.GREATER)

    def __ge__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(self, other, BinaryOperation.GREATER_EQUAL)

    def __eq__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(self, other, BinaryOperation.EQUAL)

    def __ne__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(self, other, BinaryOperation.NOT_EQUAL)

    @staticmethod
    def register_operation(operation: Operation, generator: typing.Callable,
            output: typing.Union[types.Type, typing.Callable], *args: types.Type):
        _operation_registry.setdefault(operation, []).append((args, output, generator))

    @staticmethod
    def query_operation(operation: Operation, *qargs: types.Type):
        generators = _operation_registry.get(operation, [])

        for args, output, generator in generators:
            if len(args) != len(qargs):
                continue
            if not all([a1.castable(a2) for a1, a2 in zip(args, qargs)]):
                continue
            return generator, output

        return None, None

@hidden
class Macro(Node):

    def expand(self, inputs, parent: str):
        raise NotImplementedError()

@hidden
class _UnaryOperationWrapper(Macro):

    a = Input(types.Any())
    operation = Enumeration(UnaryOperation)

    def _nodeframe(self):
        return traceback.extract_stack()[-4]

    def _init(self):
        self._generator = None

    def validate(self, **inputs):
        generator, output = Node.query_operation(self.operation, inputs["a"])

        if generator is None:
            raise ValidationException("Cannot resolve operation {} for {}".format(self.operation, inputs["a"]), node=self)

        self._generator = generator

        if isinstance(output, typing.Callable):
            return output(inputs["a"])

        return output

    def expand(self, inputs, parent: str):
        from pixelpipes.graph import GraphBuilder

        with GraphBuilder(prefix=parent) as builder:
            output = self._generator(inputs["a"][0])

            builder.rename(output, parent)

            return builder.nodes()

@hidden
class _BinaryOperationWrapper(Macro):

    a = Input(types.Any())
    b = Input(types.Any())
    operation = Enumeration(BinaryOperation)

    def _nodeframe(self):
        return traceback.extract_stack()[-4]

    def _init(self):
        self._generator = None

    def validate(self, **inputs):
        generator, output = Node.query_operation(self.operation, inputs["a"], inputs["b"])

        if generator is None:
            raise ValidationException("Cannot resolve operation {} for {}, {}".format(self.operation.name, inputs["a"], inputs["b"]), node=self)

        self._generator = generator

        if isinstance(output, typing.Callable):
            return output(inputs["a"], inputs["b"])

        return output

    def expand(self, inputs, parent: str):
        from pixelpipes.graph import GraphBuilder

        with GraphBuilder(prefix=parent) as builder:
            output = self._generator(inputs["a"][0], inputs["b"][0])

            builder.rename(output, parent)

            return builder.nodes()
