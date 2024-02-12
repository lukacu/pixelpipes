
import inspect
import typing
import threading
import numbers
import traceback
from enum import Enum, auto

from attributee import Attributee, Attribute, AttributeException, Undefined, Any, Enumeration
from attributee.object import class_fullname
from attributee.primitives import to_number, to_logical, to_string, String

from pixelpipes.types import Data

from . import types, evaluate_operation

from . import ContextFields

_CONTEXT_LOCK = threading.Condition()
_CONTEXT = threading.local()

def wrap_pybind_enum(bindenum):
    mapping = {}
    for argname in dir(bindenum):
        arg = getattr(bindenum, argname)
        if isinstance(arg, bindenum):
            mapping[arg.name] = arg

    return mapping

class NodeOperation(Enum):

    NEGATE = auto()
    INDEX = auto()
    LENGTH = auto()

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

    LOGICAL_AND = auto()
    LOGICAL_OR = auto()
    LOGICAL_NOT = auto()

class NodeException(Exception):

    def __init__(self, *args, node: typing.Optional["Node"] = None):
        super().__init__(*args)
        assert node is None or isinstance(node, Node)
        self._node = node

    @property
    def node(self):
        return self._node

    def nodestack(self):
        stack = []
        node = self._node
        while node is not None:
            stack.append(node.frame)
            node = node.origin
        return stack

    def print_nodestack(self):
        stack = self.nodestack()

        for i, frame in enumerate(stack):
            print("%d - %s" % (i, frame))

class ValidationException(NodeException):
    pass

class Input(Attribute):

    def __init__(self, reftype: types.Data, default: typing.Optional[typing.Union[str, float, int]] = None, description: typing.Optional[str] = ""):
        self._type = reftype
        assert isinstance(reftype, types.Data)
        assert default is None or isinstance(default, (int, float, bool, str)) or Reference.parse(default) is not None, "Default should be a primitive type or a reference: %s" % default
        super().__init__(default = Undefined() if default is None else default, description=description)

    def coerce(self, value, _):
        assert value is not None

        if isinstance(value, Node):
            from pixelpipes.graph import Graph
            builder = Graph.default()
            if builder is not None:
                return builder.reference(value)
            else:
                raise AttributeException("Unable to resolve node's reference")

        if isinstance(value, Reference):
            return value

        ref = Reference.parse(value)
        if ref is not None:
            return ref

        if self._type.castable(types.Token()):
            assert isinstance(value, (bool, int, float, str))
            return value

#        if self._type.castable(types.Wildcard(mindim=0, maxdim=0)):
           
        if self._type.castable(types.String()):
            return to_string(value)

        if self._type.castable(types.Float()):
            return to_number(value, conversion=float)

        if self._type.castable(types.Integer()):
            return to_number(value, conversion=int)

        if self._type.castable(types.Boolean()):
            return to_logical(value)

        raise AttributeException("Illegal value: {!s:.100}, expected {}".format(value, self._type))

    def dump(self, value):
        if isinstance(value, Reference):
            return str(value)
        return value

    def reftype(self):
        return self._type

class SeedInput(Input):
    
    def __init__(self, description=""):
        super().__init__(types.Integer(), default="@[random]", description=description)

class EnumerationInput(Input):
    
    def __init__(self, options, default=None, description=""):
        if inspect.isclass(options) and issubclass(options, Enum):
            self._mapping = options
        elif isinstance(options, typing.Mapping):
            self._mapping = options
            self._inverse = {v: k for k, v in options.items()}
        else:
            raise AttributeException("Not an enum class or dictionary")
        super().__init__(types.Integer(), default=default, description=description)

    def coerce(self, value, _):
        if isinstance(value, Node):
            from pixelpipes.graph import Graph
            builder = Graph.default()
            if builder is not None:
                return builder.reference(value)
            else:
                raise AttributeException("Unable to resolve node's reference")

        if isinstance(value, Reference):
            return value

        if value not in self._mapping:
            raise AttributeException("Illegal value '{}', valid options: '{}'".format(value, "', '".join(self._mapping.keys())))

        return self._mapping[value]

    def dump(self, value):
        if isinstance(value, Reference):
            return str(value)
        if inspect.isclass(self._mapping) and isinstance(value, self._mapping):
            return value.value
        else:
            return self._inverse[value]

_operation_registry = {}

class OperationProxy:

    def __add__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(self, other, operation=NodeOperation.ADD)

    def __radd__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(other, self, operation=NodeOperation.ADD)

    def __sub__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(self, other, operation=NodeOperation.SUBTRACT)

    def __rsub__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(other, self, operation=NodeOperation.SUBTRACT)

    def __mul__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(self, other, operation=NodeOperation.MULIPLY)

    def __rmul__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(other, self, operation=NodeOperation.MULIPLY)

    def __truediv__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(self, other, operation=NodeOperation.DIVIDE)

    def __rtruediv__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(other, self, operation=NodeOperation.DIVIDE)

    def __pow__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(self, other, operation=NodeOperation.POWER)

    def __rpow__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(other, self, operation=NodeOperation.POWER)

    def __mod__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(self, other, operation=NodeOperation.MODULO)

    def __rmod__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(other, self, operation=NodeOperation.MODULO)

    def __neg__(self):
        return _UnaryOperationWrapper(self, operation=NodeOperation.NEGATE)

    def __lt__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(self, other, operation=NodeOperation.LOWER)

    def __le__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(self, other, operation=NodeOperation.LOWER_EQUAL)

    def __gt__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(self, other, operation=NodeOperation.GREATER)

    def __ge__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(self, other, operation=NodeOperation.GREATER_EQUAL)

    def __eq__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(self, other, operation=NodeOperation.EQUAL)

    def __ne__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(self, other, operation=NodeOperation.NOT_EQUAL)

    def __getitem__(self, key):
        return _IndexOperationWrapper(self, key, operation=NodeOperation.INDEX)

    def __len__(self):
        return _UnaryOperationWrapper(self, operation=NodeOperation.INDEX)

    def __and__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(self, other, operation=NodeOperation.LOGICAL_AND)
    
    def __or__(self, other):
        other = _ensure_node(other)
        return _BinaryOperationWrapper(self, other, operation=NodeOperation.LOGICAL_OR)
    
    def __invert__(self):
        return _UnaryOperationWrapper(self, operation=NodeOperation.LOGICAL_NOT)

    @staticmethod
    def register_operation(operation: NodeOperation, generator: typing.Callable, *args: types.Data):
        _operation_registry.setdefault(operation, []).append((args, generator))

    @staticmethod
    def query_operation(operation: NodeOperation, *qargs: types.Data):
        generators = _operation_registry.get(operation, [])

        for args, generator in generators:
            if len(args) != len(qargs):
                continue
            if not all([a1.castable(a2) for a1, a2 in zip(args, qargs)]):
                continue
            return generator

        return None

class Reference(object):

    def __init__(self, ref: typing.Union[str, "Reference"]):
        if isinstance(ref, Reference):
            ref = ref.name
        if not (ref is not None and isinstance(ref, str) and ref != ""):
            raise ValidationException("Reference is undefined or illegal")
        self._ref = ref

    def __str__(self):
        return "@" + self._ref

    def __repr__(self):
        return "<@" + self._ref + ">"

    def __hash__(self):
        return hash(self.name)

    @property
    def name(self):
        return self._ref

    @staticmethod
    def parse(value):
        if isinstance(value, str) and value.startswith("@"):
            return Reference(value[1:])
        else:
            return None

    def __eq__(self, ref):
        if ref is None:
            return False
        if isinstance(ref, Reference):
            return ref.name == self.name
        if isinstance(ref, str):
            return ref == self.name
        return False

class InferredReference(Reference, OperationProxy):
    """A node reference with type or value already inferred. Used during compilation and in macro expansion."""

    def __init__(self, ref: str, data):
        if isinstance(ref, Reference):
            ref = ref.name
        super().__init__(ref)
        self._data = data

    @property
    def type(self):
        if isinstance(self._data, types.Data):
            return self._data
        else:
            shape = self._data.shape()
            return types.Token(shape[0], *shape[1:])

def hidden(node_class):
    node_class.__hidden_base = node_class
    return node_class

def _ensure_node(value):
    from .graph import Graph
    if not Graph.has_default():
        raise ValueError("Unable to use node operation magic without a context graph")

    if isinstance(value, Reference):
        return value
    if isinstance(value, Node):
        return value
    elif isinstance(value, numbers.Number):
        from .graph import Constant
        return Constant(value=value)
    else:
        raise ValueError("Value is not a node or a number")

@hidden
class Node(Attributee, OperationProxy):
    """Base class for all nodes in a computation graph.
    """

    def __init__(self, *args, _name: str = None, _auto: bool = True, _origin: "Node" = None, **kwargs):
        #Node constructor, please do not overload it directly, override _init method to add custom 
        #initialization code.
        #
        #Args:
        #    _name (str, optional): Internal parameter used for context graph builder. Defaults to None.
        #    _auto (bool, optional): Should a node automatically be added to a context builder. Defaults to True.
        
        super().__init__(*args, **kwargs)

        self._inputs_cache = None

        if _auto:
            from pixelpipes.graph import Graph
            Graph.add_default(self, _name)

        self._origin = _origin
        self._source = self._nodeframe()
        self._init()

    def _nodeframe(self):
        return traceback.extract_stack()[-3]

    def _init(self):
        pass

    @classmethod
    def name(cls):
        if hasattr(cls, "node_name"):
            return getattr(cls, "node_name")
        else:
            return cls.__name__

    @classmethod
    def hidden(cls):
        if hasattr(cls, "__hidden_base"):
            return getattr(cls, "__hidden_base") == cls
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
            assert k in config, "Key %s not in config" % k
            config[k] = v
        return config

    def duplicate(self, _name=None, _origin=None, **inputs):
        config = self._merge_config(self.dump(), inputs)
        double = self.__class__(_auto=False, _name=_name, _origin=_origin, **config)
        double._origin = self._origin
        double._source = self._source
        return double

    def evaluate(self, **inputs):
        input_types = self.get_inputs()
        assert len(inputs) == len(input_types)
        for input_name, input_type in input_types:
            if not input_name in inputs:
                raise ValidationException("Input '{}' not available".format(input_name), node=self)
            if not input_type.castable(inputs[input_name]):
                raise ValidationException("{}: input '{}' cannot convert {} to {}".format(
                    class_fullname(self), input_name, inputs[input_name], input_type), node=self)
        try:
            return self._evaluate([inputs[name] for name, _ in input_types])
        except TypeError as te:
            raise ValidationException("Inferrence failed: {}".format(te), node=self)

    def input_types(self):
        return [i for _, i in self.get_inputs()]

    def input_values(self):
        return [getattr(self, name) for name, _ in self.get_inputs()]

    def input_names(self):
        return [name for name, _ in self.get_inputs()]

    def _evaluate(self, inputs) -> types.Data:
        raise types.TypeException("Type inferrence must be implemented: {}".format(class_fullname(self)))

    def get_inputs(self):
        if self._inputs_cache is not None:
            return self._inputs_cache
        references = []
        for name, attr in self.attributes().items():
            if isinstance(attr, Input):
                references.append((name, attr.reftype()))
        self._inputs_cache = references
        return references

    @property
    def origin(self):
        """Only relevant for nodes generated during compilation, makes it easier to 
        track original node from the source graph. Returns None in other cases.
        """
        return self._origin

    def __hash__(self):
        return id(self)

@hidden
class Operation(Node):
    """Base class of all atomic nodes that generate pipeline operations
    """

    def operation(self) -> typing.Tuple:
        """Generates a operation construction data that is passed to the library

        Returns:
            typing.Tuple: A tuple of library arguments, the first one being the name of the operation and the rest its construction
            arguments.
        """
        raise NodeException("Node not converable to operation: " + str(self), node=self)

    def _evaluate(self, inputs) -> types.Data:
        data = self.operation()
        try:
            return evaluate_operation(data[0], inputs, data[1:])
        except Exception as e:
            strinputs = ", ".join([str(i) for i in inputs])
            raise NodeException("Error during operation evaluation ({}): {} - {}".format(self.__class__.__name__, e, strinputs), node=self)


@hidden
class Macro(Node):

    def expand(self, **inputs):
        raise NotImplementedError()

    def evaluate(self, **inputs):
        # Macro inferrence will never be used, this is just a placeholder
        return None

@hidden
class _OperationWrapper(Macro):

    operation = Enumeration(NodeOperation)

    def _nodeframe(self):
        return traceback.extract_stack()[-4]

@hidden
class _UnaryOperationWrapper(_OperationWrapper):

    a = Input(types.Anything())

    def expand(self, a):
        generator = Node.query_operation(self.operation, a.type)
        if generator is None:
            raise ValidationException("Cannot resolve operation {} for {}".format(self.operation, a.type), node=self)
        return generator(a)

@hidden
class _BinaryOperationWrapper(_OperationWrapper):

    a = Input(types.Anything())
    b = Input(types.Anything())

    def _nodeframe(self):
        return traceback.extract_stack()[-4]

    def expand(self, a, b):
        generator = Node.query_operation(self.operation, a.type, b.type)

        if generator is None:
            raise ValidationException("Cannot resolve operation {} for {}, {}".format(self.operation.name, a.type, b.type), node=self)

        return generator(a, b)

@hidden
class _IndexOperationWrapper(_OperationWrapper):
    """This is a specialized wrapper for object indexing operations that does internal
    """

    container = Input(types.Anything())
    index = Input(types.Anything())

    def _init(self):
        # Store raw index after initialization so that we can use it later for operations that do not
        # support dynamic indexing (e.g. resources)
        self._rawindex = self.index

    def expand(self, container, index):
        generator = Node.query_operation(self.operation, container.type, index.type)
        if generator is None:
            raise ValidationException("Cannot resolve operation {} for {}, {}".format(self.operation.name, container.type, index.type), node=self)
        
        if isinstance(self._rawindex, (Node, Reference)):
            return generator(container, index)
        else:
            return generator(container, self._rawindex)

class Graph(object):

    def __init__(self, prefix: typing.Optional[typing.Union[str, Reference]] = ""):
        import bidict
        from .utilities import Counter

        self._nodes = bidict.bidict()
        self._count = Counter()
        self._prefix = prefix if isinstance(prefix, str) else prefix.name
        self._parent = None

    def subgraph(self, prefix: typing.Optional[typing.Union[str, Reference]] = "") -> "Graph":
        graph = Graph(prefix)
        graph._parent = self
        return graph

    def commit(self):
        if self._parent is None:
            raise NodeException("No parent graph to commit to")
        self._parent += self

    def copy(self):
        graph = Graph()
        graph += self
        return graph

    @staticmethod
    def has_default():
        with _CONTEXT_LOCK:
            builders = getattr(_CONTEXT, "builders", [])
            return len(builders) > 0
            
    @staticmethod
    def default() -> "Graph":
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
            while True:
                name = "node%d" % self._count()
                if self._prefix != "":
                    name = self._prefix + "." + name
                if name in self._nodes or (self._parent is not None and name in self._parent._nodes):
                    continue
                break
        else:
            if name.startswith("."):
                if name == ".":
                    name = self._prefix
                else:
                    name = self._prefix + name
            if name in self._nodes or self._parent and node in self._parent._nodes:
                raise NodeException("Name already exists: {}".format(name), node=node)

        self._nodes[name] = node
        return Reference(name)

    def remove(self, node: typing.Union[Node, Reference]):
        if isinstance(node, Node):
            node = self._nodes.inverse[node]
        if node is None or node not in self._nodes:
            return
        del self._nodes[node]

    def replace(self, oldnode: typing.Union[Node, Reference], newnode: Node):
        if isinstance(oldnode, Node):
            name = self.reference(oldnode)
        else:
            name = oldnode
        self.remove(oldnode)
        self.add(newnode, name)

    def reference(self, node: Node):
        assert isinstance(node, Node)
        name = self._nodes.inverse[node]
        return Reference(name)

    def nodes(self):
        return dict(**self._nodes)

    def __iter__(self):
        for k, v in self._nodes.items():
            yield Reference(k), v

    def __getitem__(self, ref: Reference):
        return self._nodes[ref.name]

    def pipeline(self, fixedout=False, variables = None, output=None):
        from pixelpipes.compiler import Compiler
        
        return Compiler(fixedout=fixedout).build(self, variables=variables, output=output)

    def __contains__(self, node: typing.Union[Node, Reference]):
        if isinstance(node, Reference):
            return node.name in self._nodes
        if isinstance(node, Node):
            return node in self._nodes.inverse
        return False

    def __add__(self, element):
        if isinstance(element, Node):
            self.add(element)
        elif isinstance(element, Graph) and element != self:
            for name, node in element.nodes().items():
                self.add(node, name)
        else:
            raise RuntimeError("Illegal value")
        return self

    def __len__(self):
        return len(self._nodes)

def _list_type(source):
    if all(isinstance(x, bool) for x in source):
        return types.Boolean()
    if all(isinstance(x, (int, bool)) for x in source):
        return types.Integer()
    if all(isinstance(x, (bool, int, float)) for x in source):
        return types.Float()
    else:
        typ = types.Wildcard()
        for x in source:
            typ = typ.common(Constant.resolve_type(x))
        return typ

class Constant(Operation):
    """ Generates a constant in the pipeline """

    value = Any()

    def operation(self):
        return "constant", self.value

    @staticmethod
    def resolve_type(value) -> types.Data:
        import numpy as np

        if isinstance(value, bool):
            return types.Boolean()
        elif isinstance(value, int):
            return types.Integer()
        elif isinstance(value, float):
            return types.Float()
        elif isinstance(value, str):
            return types.List("char", len(value))
        elif isinstance(value, np.ndarray):
            return types.Token(value.dtype, *value.shape)
        elif isinstance(value, list):
            return _list_type(value).push(length=len(value))
        raise types.TypeException("Unsupported constant value: {!s:.100}".format(value))

    def key(self):
        if isinstance(self.value, bool):
            return ("bool", hash(self.value))
        elif isinstance(self.value, int):
            return ("int", hash(self.value))
        elif isinstance(self.value, float):
            return ("float", hash(self.value))
        elif isinstance(self.value, str):
            return ("str", hash(self.value))
        return None

class SampleIndex(Operation):
    """Returns current sample index. This information can be used instead of random seed
    to initialize random generators where sequential consistentcy is required.
    """

    def operation(self):
        return "context", ContextFields["SampleIndex"]

class RandomSeed(Operation):
    """Returns a pseudo-random number, useful for initializing pseudo-random operations. The seed
    itself is sampled from a pseudo-random generator that produces the same sequence of seeds for
    a specific position in the data sequence. This is the corner-stone of repeatability of the pipeline. 
    """

    def operation(Operation):
        return "context", ContextFields["RandomSeed"]

class Debug(Operation):
    """Debug operation enables low-level terminal output of the content that is provided to it. 
    The token content will usually not be printed entierly, only its shape, the value will only be displayed
    for simple scalar types as well as strings.
    
    Note that tese nodes will be passed to the pipeline only if the compiler is configered with debug flag,
    otherwise they will be stripped from the graph. 
    """

    source = Input(types.Wildcard(), description="Result of which node to print")
    prefix = String(default="", description="String that is prepended to the output")

    def operation(self):
        return "debug", self.prefix
    
    def _evaluate(self, inputs) -> Data:
        return inputs[0]
 
class Output(Operation):
    """Output node that accepts a single input, enables outputting tokens from the final pipeline. Tokens
    are returned as a tuple, their order is determined by the order of adding output nodes to the graph. Additionally
    you may also label outputs with non-unique lables that can be used to resolve outputs.
    """

    output = Input(types.Wildcard(), description="Output token")

    label = String(default="default", description="Nonunique label of the output")

    def operation(self):
        return "output", self.label

class ReadFile(Operation):
    """Read file from disk to memory buffer. File is read in binary mode."""

    filename = Input(types.String(), description="Path to the image file")

    def operation(self):
        return "read_file",

def outputs(*inputs, label="default"):
    for i in inputs:
        Output(output=i, label=label)

@hidden
class Copy(Node):

    source = Input(types.Wildcard())

    def _evaluate(self, inputs):
        return inputs[0]

