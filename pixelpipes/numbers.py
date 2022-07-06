
from attributee import Boolean

from . import types
from .graph import Macro, Operation, Node, Input, SeedInput, hidden, NodeOperation, Constant

class SampleUnform(Operation):
    """Samples random value between min and max value."""

    min = Input(types.Float(), description="Minimun value")
    max = Input(types.Float(), description="Maximum value")
    seed = SeedInput()

    def infer(self, min, max, seed) -> types.Data:
        return types.Float()

    def operation(self):
        return "random_uniform",

class SampleNormal(Operation):
    """Samples values between from normal distribution.
    """

    mean = Input(types.Float(), default=0, description="Mean value of normal distribution")
    sigma = Input(types.Float(), default=1, description="Standard deviation")
    seed = SeedInput()

    def infer(self, mean, sigma, seed) -> types.Data:
        return types.Float()

    def operation(self):
        return "random_normal",

class RandomBoolean(Macro):
    """Samples a boolean value with equal probability
    """

    seed = SeedInput()

    def expand(self, seed):
        return SampleUnform(0, 1, seed) >= 0.5

class Round(Operation):
    """Round number to closest integer and convert to integer type."""

    source = Input(types.Float(), description="Number to be rounded")

    def infer(self, source):
        return types.Integer()

    def operation(self):
        return "numbers_round",

class Floor(Operation):
    """Floor number and convert to integer."""

    source = Input(types.Float(), description="Number to be rounded")

    def infer(self, source):
        return types.Integer()

    def operation(self):
        return "numbers_floor",

class Ceil(Operation):
    """Ceil number and convert to integer.
    """

    source = Input(types.Float(), description="Number on which ceil operation is performed")

    def infer(self, source):
        return types.Integer()

    def operation(self):
        return "numbers_ceil",

@hidden
class _BinaryOperator(Operation):

    a = Input(types.Float(), description="First operand")
    b = Input(types.Float(), description="Second operand")

    def infer(self, a, b):
        return a.common(b)

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

    def infer(self, **inputs):
        return types.Boolean()        

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

Node.register_operation(NodeOperation.ADD, Add, types.Float(), types.Float())
Node.register_operation(NodeOperation.SUBTRACT, Subtract, types.Float(), types.Float())
Node.register_operation(NodeOperation.MULIPLY, Multiply, types.Float(), types.Float())
Node.register_operation(NodeOperation.DIVIDE, Divide, types.Float(), types.Float())
Node.register_operation(NodeOperation.POWER, Power, types.Float(), types.Float())
Node.register_operation(NodeOperation.MODULO, Modulo, types.Float(), types.Float())

Node.register_operation(NodeOperation.NEGATE, lambda x: Multiply(x, Constant(-1)), types.Float())

Node.register_operation(NodeOperation.GREATER, Greater, types.Float(), types.Float())
Node.register_operation(NodeOperation.GREATER_EQUAL, GreaterEqual, types.Float(), types.Float())
Node.register_operation(NodeOperation.LOWER, Lower, types.Float(), types.Float())
Node.register_operation(NodeOperation.LOWER_EQUAL, LowerEqual, types.Float(), types.Float())
Node.register_operation(NodeOperation.EQUAL, Equal, types.Float(), types.Float())
Node.register_operation(NodeOperation.NOT_EQUAL, NotEqual, types.Float(), types.Float())

def _register_tensor_operation(operation, generator):
    Node.register_operation(operation, generator, types.Wildcard(mindim=1), types.Wildcard(mindim=1))
    Node.register_operation(operation, generator, types.Wildcard(mindim=1), types.Float())
    Node.register_operation(operation, generator, types.Float(), types.Wildcard(mindim=1))

def _tensor_piecewise_infer(a: types.Token, b: types.Token):
    a = a.squeeze()
    b = b.squeeze()

    #for i in range(max(a.rank, b.rank)):
    #    if a[i] is not None and b[i] is not None and a[i] != b[i]:
    #        raise types.TypeException("Size mismatch")

    #if a.element is not None and b.element is not None and a.element != b.element:
    #    raise types.TypeException("Element mismatch, {} and {}  ".format(a.element, b.element))


    return a.common(b)

class TensorAdd(Operation):

    a = Input(types.Wildcard(), description="First operand")
    b = Input(types.Wildcard(), description="Second operand")
    saturate = Boolean(default=False, description="Saturate cast")

    def operation(self):
        if self.saturate:
            return "tensor_add_saturate",
        return "tensor_add",

    def infer(self, a, b):
        return _tensor_piecewise_infer(a, b)

_register_tensor_operation(NodeOperation.ADD, TensorAdd)

class TensorSubtract(Operation):
    """Subtracts two images with same size and number of channels or an image and a number.
    """

    a = Input(types.Wildcard(), description="First operand")
    b = Input(types.Wildcard(), description="Second operand")
    saturate = Boolean(default=False, description="Saturate cast")

    def operation(self):
        if self.saturate:
            return "tensor_subtract_saturate",
        return "tensor_subtract",

    def infer(self, a, b):
        return _tensor_piecewise_infer(a, b)

_register_tensor_operation(NodeOperation.SUBTRACT, TensorSubtract)

class TensorMultiply(Operation):
    """Multiplies image with another image or scalar (per-element multiplication).
    """

    a = Input(types.Wildcard(), description="First operand")
    b = Input(types.Wildcard(), description="Second operand")
    saturate = Boolean(default=False, description="Saturate cast")

    def operation(self):
        if self.saturate:
            return "tensor_multiply_saturate",
        return "tensor_multiply",

    def infer(self, a, b):
        return _tensor_piecewise_infer(a, b)

_register_tensor_operation(NodeOperation.MULIPLY, TensorMultiply)

class TensorDivide(Operation):
    """Divides image with another image or scalar (per-element multiplication).
    """

    a = Input(types.Wildcard(), description="First operand")
    b = Input(types.Wildcard(), description="Second operand")
    saturate = Boolean(default=False, description="Saturate cast")

    def operation(self):
        if self.saturate:
            return "tensor_divide_saturate",
        return "tensor_divide",

    def infer(self, a, b):
        return _tensor_piecewise_infer(a, b)

_register_tensor_operation(NodeOperation.DIVIDE, TensorDivide)