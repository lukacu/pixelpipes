
from attributee import String, List
from attributee.primitives import Number

from ..node import Node, NodeException, Input, hidden
from .. import types
from .. import pp_py

def make_operation(name, *args):
    try:
        return pp_py.make(name, *args)
    except (RuntimeError, ValueError) as e:
        raise ValueError("Cannot create operation %s: %s" % (name, e))

def load_operation_module(name):
    return pp_py.load(name)

# TODO: make all dicts contant
def get_enum(identifier):
    return pp_py.enum(identifier)

def create_pipeline():
    return pp_py.Pipeline()

ContextFields = get_enum("context")
ComparisonOperations = get_enum("comparison")
LogicalOperations = get_enum("logical")
ArithmeticOperations = get_enum("arithmetic")
class Constant(Node):

    node_name = "Constant"
    node_description = "Outputs a constant number"
    node_category = "numeric"

    value = Number(default=0)

    def operation(self):
        return "_constant", self.value

    def _output(self) -> types.Type:
        return Constant.resolve_type(self.value)

    @staticmethod
    def resolve_type(value) -> types.Type:
        if isinstance(value, int):
            return types.Integer(value)
        else:
            return types.Float(value)

    def key(self):
        typ = Constant.resolve_type(self.value)
        if isinstance(typ, types.Integer):
            return ("int", self.value)
        else:
            return ("float", self.value)


class SampleNumber(Node):

    def _output(self):
        return types.Integer()

    def operation(self):
        return "_context", ContextFields["SampleIndex"]

class DebugOutput(Node):

    source = Input(types.Primitive())
    prefix = String(default="")

    def validate(self, **inputs):
        super().validate(**inputs)
        return inputs["source"]

    def operation(self):
        return "_debug", self.prefix
 
class Output(Node):

    outputs = List(Input(types.Primitive()))

    identifier = String(default="default")

    def _output(self) -> types.Type:
        return None

    def get_inputs(self):
        return [(str(i), types.Any()) for i, _ in enumerate(self.outputs)]

    def input_values(self):
        return [self.outputs[int(name)] for name, _ in self.get_inputs()]

    def operation(self):
        return "_output",

    def duplicate(self, **inputs):
        config = self.dump()
        for k, v in inputs.items():
            i = int(k)
            assert i >= 0 and i < len(config["outputs"])
            config["outputs"][i] = v
        return self.__class__(**config)

@hidden
class Copy(Node):

    source = Input(types.Primitive())

    def validate(self, **inputs):
        super().validate(**inputs)
        return inputs["source"]

    def operation(self):
        raise NodeException("Copy node should be removed, it does not do anything", node=self)
