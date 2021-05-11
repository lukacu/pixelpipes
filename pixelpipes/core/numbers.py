from attributee import List, Number, Enumeration

from . import ComparisonOperations

from ..core import Constant
from ..node import Node, Input, NodeException, hidden, BinaryOperation, UnaryOperation
from .. import types

class UniformDistribution(Node):

    node_name = "Uniform distribution"
    node_description = "Samples values between min and max"
    node_category = "numeric"

    min = Input(types.Float())
    max = Input(types.Float())

    def _output(self) -> types.Type:
        return types.Float()

    def operation(self):
        return "random_uniform",

class NormalDistribution(Node):

    node_name = "Normal distribution"
    node_description = "Samples values between from normal distribution"
    node_category = "numeric"

    mean = Input(types.Float(), default=0)
    sigma = Input(types.Float(), default=1)

    def _output(self) -> types.Type:
        return types.Float()

    def operation(self):
        return "random_normal",

class Round(Node):

    node_name = "Round"
    node_description = "Round number and convert to integer"
    node_category = "numeric"

    source = Input(types.Float())

    def validate(self, **inputs):
        super().validate(**inputs)

        if inputs["source"].value is not None:
            return types.Integer(round(inputs["source"].value))
        else:
            return types.Integer()

    def operation(self):
        return "numbers_round",

class Floor(Node):

    source = Input(types.Float())

    def validate(self, **inputs):
        super().validate(**inputs)

        if inputs["source"].value is not None:
            return types.Integer(round(inputs["source"].value))
        else:
            return types.Integer()

    def operation(self):
        return "numbers_floor",

class Ceil(Node):

    source = Input(types.Float())

    def validate(self, **inputs):
        super().validate(**inputs)

        if inputs["source"].value is not None:
            return types.Integer(round(inputs["source"].value))
        else:
            return types.Integer()

    def operation(self):
        return "numbers_ceil",

@hidden
class _BinaryOperator(Node):

    a = Input(types.Number())
    b = Input(types.Number())

    def validate(self, **inputs):
        super().validate(**inputs)

        integer = isinstance(inputs["a"], types.Integer) and isinstance(inputs["b"], types.Integer)

        if inputs["a"].value is not None and inputs["b"].value is not None:
            return self._compute(inputs["a"].value, inputs["b"].value)

        return types.Integer() if integer else types.Float()

    def _compute(self, a, b):
        raise NotImplementedError()

class Add(_BinaryOperator):

    def operation(self):
        return "numbers_add",

    def _compute(self, a, b):
        return Constant.resolve_type(a + b)

class Multiply(_BinaryOperator):

    def operation(self):
        return "numbers_multiply",

    def _compute(self, a, b):
        return Constant.resolve_type(a * b)

class Subtract(_BinaryOperator):

    def operation(self):
        return "numbers_subtract",

    def _compute(self, a, b):
        return Constant.resolve_type(a - b)

class Divide(_BinaryOperator):

    def operation(self):
        return "numbers_divide",

    def _compute(self, a, b):
        return types.Float(a / b)

class Power(_BinaryOperator):

    def operation(self):
        return "numbers_power",

    def _compute(self, a, b):
        return Constant.resolve_type(a ** b)

class Modulo(_BinaryOperator):

    a = Input(types.Integer())
    b = Input(types.Integer())

    def operation(self):
        return "numbers_modulo",

    def _compute(self, a, b):
        return types.Integer(a % b)

@hidden
class _ComparisonOperation(_BinaryOperator):

    def validate(self, **inputs):
        inferred = super().validate(**inputs)
        if inferred.value is not None:
            return types.Integer(inferred.value != 0)

        return types.Integer()
class Greater(_ComparisonOperation):

    def operation(self):
        return "comparison", ComparisonOperations["GREATER"]

    def _compute(self, a, b):
        return types.Integer(a > b)

class Lower(_ComparisonOperation):

    def operation(self):
        return "comparison", ComparisonOperations["LOWER"]

    def _compute(self, a, b):
        return types.Integer(a < b)

class GreaterEqual(_ComparisonOperation):

    def operation(self):
        return "comparison", ComparisonOperations["GREATER_EQUAL"]

    def _compute(self, a, b):
        return types.Integer(a >= b)

class LowerEqual(_ComparisonOperation):

    def operation(self):
        return "comparison", ComparisonOperations["LOWER_EQUAL"]

    def _compute(self, a, b):
        return types.Integer(a <= b)

class Equal(_ComparisonOperation):

    def operation(self):
        return "comparison", ComparisonOperations["EQUAL"]

    def _compute(self, a, b):
        return types.Integer(a == b)

class Maximum(_BinaryOperator):

    def operation(self):
        return "numbers_max",

    def _compute(self, a, b):
        return Constant.resolve_type(max(a, b))

class Minimum(_BinaryOperator):

    def operation(self):
        return "numbers_min", 

    def _compute(self, a, b):
        return Constant.resolve_type(min(a, b))

class Threshold(Node):

    threshold = Number()
    comparison = Enumeration(ComparisonOperations)
    source = Input(types.Number())

    def _output(self):
        return types.Integer()

    def operation(self):
        return "threshold", self.threshold, self.comparison

class _ThresholdsComparison(Node):

    thresholds = List(Number())
    comparison = List(Enumeration(ComparisonOperations))
    inputs = List(Input(types.Number()))

    def _init(self):
        if len(self.inputs) == 0:
            raise NodeException("No inputs provided", node=self)

        if len(self.inputs) != len(self.thresholds) or len(self.inputs) != len(self.comparison):
            raise NodeException("Number of inputs and conditions does not match", node=self)

    def input_values(self):
        return [self.inputs[int(name)] for name, _ in self.get_inputs()]

    def get_inputs(self):
        return [(str(k), types.Number()) for k, _ in enumerate(self.inputs)]

    def duplicate(self, **inputs):
        config = self.dump()
        for k, v in inputs.items():
            i = int(k)
            assert i >= 0 and i < len(config["inputs"])
            config["inputs"][i] = v
        return self.__class__(**config)

    def _output(self):
        return types.Integer()

class ThresholdsConjunction(_ThresholdsComparison):

    def operation(self):
        return engine.ThresholdsConjunction(self.thresholds, self.comparison)

class ThresholdsDisjunction(_ThresholdsComparison):

    def operation(self):
        return engine.ThresholdsDisjunction(self.thresholds, self.comparison)


def _numeric_unary_generator(proto, a):
    refa, typa = tuple(a)
    if types.Number().castable(typa):
        return proto(refa)

def _numeric_binary_generator(proto, a, b):
    refa, typa = tuple(a)
    refb, typb = tuple(b)
    if types.Number().castable(typa) and types.Number().castable(typb):
        return proto(refa, refb)

def _infer_type(t1, t2 = None):
    if t2 is None:
        return t1

    integer = isinstance(t1, types.Integer) and isinstance(t2, types.Integer)

    return types.Integer() if integer else types.Float()

Node.register_operation(BinaryOperation.ADD, Add, _infer_type, types.Number(), types.Number())
Node.register_operation(BinaryOperation.SUBTRACT, Subtract, _infer_type, types.Number(), types.Number())
Node.register_operation(BinaryOperation.MULIPLY, Multiply, _infer_type, types.Number(), types.Number())
Node.register_operation(BinaryOperation.DIVIDE, Divide, _infer_type, types.Number(), types.Number())
Node.register_operation(BinaryOperation.POWER, Power, _infer_type, types.Number(), types.Number())
Node.register_operation(BinaryOperation.MODULO, Modulo, _infer_type, types.Number(), types.Number())

Node.register_operation(UnaryOperation.NEGATE, lambda x: Multiply(x, Constant(-1)), _infer_type, types.Number())

def _infer_logical(a, b):
    return types.Integer()

Node.register_operation(BinaryOperation.GREATER, Greater, _infer_logical, types.Number(), types.Number())
Node.register_operation(BinaryOperation.GREATER_EQUAL, GreaterEqual, _infer_logical, types.Number(), types.Number())
Node.register_operation(BinaryOperation.LOWER, Lower, _infer_logical, types.Number(), types.Number())
Node.register_operation(BinaryOperation.LOWER_EQUAL, LowerEqual, _infer_logical, types.Number(), types.Number())