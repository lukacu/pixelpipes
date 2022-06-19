from attributee import List, Number, Enumeration

from . import ComparisonOperations, types
from .graph import Node, Input, NodeException, SeedInput, hidden, BinaryOperation, UnaryOperation, Constant

class UniformDistribution(Node):
    """Uniform distribution

    Samples values between min and max.

    Inputs:
        - min: Minimun value
        - max: Maximum value

    Category: numeric
    """

    min = Input(types.Float())
    max = Input(types.Float())
    seed = SeedInput()

    def _output(self) -> types.Type:
        return types.Float()

    def operation(self):
        return "random_uniform",

class NormalDistribution(Node):
    """Normal distribution

    Samples values between from normal distribution.

    Inputs:
        - mean: Mean value of normal distribution
        - sigma: Standard deviation

    Category: numeric
    """

    mean = Input(types.Float(), default=0)
    sigma = Input(types.Float(), default=1)
    seed = SeedInput()

    def _output(self) -> types.Type:
        return types.Float()

    def operation(self):
        return "random_normal",

class Round(Node):
    """Round

    Round number to closest integer and convert to integer type.

    Inputs:
        - source: Number to be rounded

    Category: numeric
    """

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
    """Floor

    Floor number and convert to integer.

    Inputs:
        - source: Number on which floor operation is performed

    Category: numeric
    """

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
    """Ceil

    Ceil number and convert to integer.

    Inputs:
        - source: Number on which ceil operation is performed

    Category: numeric
    """

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

        return types.Integer() if integer else types.Float()

class Add(_BinaryOperator):

    def operation(self):
        return "numbers_add",

class Multiply(_BinaryOperator):

    def operation(self):
        return "numbers_multiply",

class Subtract(_BinaryOperator):

    def operation(self):
        return "numbers_subtract",

class Divide(_BinaryOperator):

    def operation(self):
        return "numbers_divide",


class Power(_BinaryOperator):

    def operation(self):
        return "numbers_power",

class Modulo(_BinaryOperator):

    a = Input(types.Integer())
    b = Input(types.Integer())

    def operation(self):
        return "numbers_modulo",

@hidden
class _ComparisonOperation(_BinaryOperator):

    def validate(self, **inputs):
        inferred = super().validate(**inputs)
        if inferred.value is not None:
            return types.Integer(inferred.value != 0)

        return types.Integer()


class Greater(_ComparisonOperation):

    def operation(self):
        return "compare_greater",

class Lower(_ComparisonOperation):

    def operation(self):
        return "compare_less",

class GreaterEqual(_ComparisonOperation):

    def operation(self):
        return "compare_greater_equal",

class NotEqual(_ComparisonOperation):

    def operation(self):
        return "compare_not_equal",

class LowerEqual(_ComparisonOperation):

    def operation(self):
        return "compare_less_equal",

class Equal(_ComparisonOperation):

    def operation(self):
        return "compare_equal",

class Maximum(_BinaryOperator):

    def operation(self):
        return "numbers_max",

class Minimum(_BinaryOperator):

    def operation(self):
        return "numbers_min", 

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
Node.register_operation(BinaryOperation.EQUAL, Equal, _infer_logical, types.Number(), types.Number())
Node.register_operation(BinaryOperation.NOT_EQUAL, NotEqual, _infer_logical, types.Number(), types.Number())