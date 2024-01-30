
from attributee import Boolean, List

from . import types
from .graph import Macro, Operation, Node, Input, SeedInput, NodeOperation, Constant, NodeException

class SampleUnform(Operation):
    """Samples random value between min and max value."""

    min = Input(types.Float(), description="Minimun value")
    max = Input(types.Float(), description="Maximum value")
    seed = SeedInput()

    def operation(self):
        return "sample_uniform",

class SampleNormal(Operation):
    """Samples values between from normal distribution.
    """

    mean = Input(types.Float(), default=0, description="Mean value of normal distribution")
    sigma = Input(types.Float(), default=1, description="Standard deviation")
    seed = SeedInput()

    def operation(self):
        return "sample_normal",

class SampleBernoulli(Operation):

    p = Input(types.Float(), description="Probability of True")
    seed = SeedInput()

    def operation(self):
        return "sample_bernoulli",

class SampleBinomial(Operation):

    p = Input(types.Float(), description="Probability of True")
    n = Input(types.Integer(), description="Number of samples")
    seed = SeedInput()

    def operation(self):
        return "sample_binomial",

class RandomBoolean(Macro):
    """Samples a boolean value with equal probability
    """

    seed = SeedInput()

    def expand(self, seed):
        return SampleUnform(0, 1, seed) >= 0.5

class Add(Operation):

    a = Input(types.Wildcard(), description="First operand")
    b = Input(types.Wildcard(), description="Second operand")
    saturate = Boolean(default=False, description="Saturate cast")

    def operation(self):
        if self.saturate:
            return "add_saturate",
        return "add",

class Multiply(Operation):

    a = Input(types.Wildcard(), description="First operand")
    b = Input(types.Wildcard(), description="Second operand")
    saturate = Boolean(default=False, description="Saturate cast")

    def operation(self):
        if self.saturate:
            return "multiply_saturate",
        return "multiply",

class Subtract(Operation):

    a = Input(types.Wildcard(), description="First operand")
    b = Input(types.Wildcard(), description="Second operand")
    saturate = Boolean(default=False, description="Saturate cast")

    def operation(self):
        if self.saturate:
            return "subtract_saturate",
        return "subtract",

class Divide(Operation):

    a = Input(types.Wildcard(), description="First operand")
    b = Input(types.Wildcard(), description="Second operand")
    saturate = Boolean(default=False, description="Saturate cast")

    def operation(self):
        if self.saturate:
            return "divide",
        return "divide",


class Power(Operation):

    a = Input(types.Wildcard(), description="First operand")
    b = Input(types.Wildcard(), description="Second operand")

    def operation(self):
        return "power",

class Sqrt(Operation):

    a = Input(types.Wildcard(), description="Input value")

    def operation(self):
        return "sqrt",

class Modulo(Operation):

    a = Input(types.Wildcard(), description="First operand")
    b = Input(types.Wildcard(), description="Second operand")

    def operation(self):
        return "modulo",

class Greater(Operation):

    a = Input(types.Wildcard(), description="First operand")
    b = Input(types.Wildcard(), description="Second operand")

    def operation(self):
        return "greater",

class Lower(Operation):

    a = Input(types.Wildcard(), description="First operand")
    b = Input(types.Wildcard(), description="Second operand")

    def operation(self):
        return "less",

class GreaterEqual(Operation):

    a = Input(types.Wildcard(), description="First operand")
    b = Input(types.Wildcard(), description="Second operand")

    def operation(self):
        return "greater_equal",

class NotEqual(Operation):

    a = Input(types.Wildcard(), description="First operand")
    b = Input(types.Wildcard(), description="Second operand")

    def operation(self):
        return "not_equal",

class LowerEqual(Operation):

    a = Input(types.Wildcard(), description="First operand")
    b = Input(types.Wildcard(), description="Second operand")

    def operation(self):
        return "less_equal",

class Equal(Operation):

    a = Input(types.Wildcard(), description="First operand")
    b = Input(types.Wildcard(), description="Second operand")

    def operation(self):
        return "equal",

class Maximum(Operation):

    a = Input(types.Wildcard(), description="First operand")
    b = Input(types.Wildcard(), description="Second operand")

    def operation(self):
        return "max",

class Minimum(Operation):

    a = Input(types.Wildcard(), description="First operand")
    b = Input(types.Wildcard(), description="Second operand")

    def operation(self):
        return "min", 

Node.register_operation(NodeOperation.ADD, Add, types.Wildcard(), types.Wildcard())
Node.register_operation(NodeOperation.SUBTRACT, Subtract, types.Wildcard(), types.Wildcard())
Node.register_operation(NodeOperation.MULIPLY, Multiply, types.Wildcard(), types.Wildcard())
Node.register_operation(NodeOperation.DIVIDE, Divide, types.Wildcard(), types.Wildcard())
Node.register_operation(NodeOperation.POWER, Power, types.Wildcard(), types.Wildcard())
Node.register_operation(NodeOperation.MODULO, Modulo, types.Wildcard(), types.Wildcard())

Node.register_operation(NodeOperation.NEGATE, lambda x: Multiply(x, Constant(-1)), types.Wildcard())

Node.register_operation(NodeOperation.GREATER, Greater, types.Wildcard(), types.Wildcard())
Node.register_operation(NodeOperation.GREATER_EQUAL, GreaterEqual, types.Wildcard(), types.Wildcard())
Node.register_operation(NodeOperation.LOWER, Lower, types.Wildcard(), types.Wildcard())
Node.register_operation(NodeOperation.LOWER_EQUAL, LowerEqual, types.Wildcard(), types.Wildcard())
Node.register_operation(NodeOperation.EQUAL, Equal, types.Wildcard(), types.Wildcard())
Node.register_operation(NodeOperation.NOT_EQUAL, NotEqual, types.Wildcard(), types.Wildcard())

# Rounding

class Round(Operation):
    """Round number to closest integer and convert to integer type."""

    source = Input(types.Wildcard(), description="Number to be rounded")

    def operation(self):
        return "round",

class Floor(Operation):
    """Floor number and convert to integer."""

    source = Input(types.Wildcard(), description="Number to be rounded")

    def operation(self):
        return "floor",

class Ceil(Operation):
    """Ceil number and convert to integer.
    """

    source = Input(types.Wildcard(), description="Number on which ceil operation is performed")

    def operation(self):
        return "ceil",

# Trigonometric functions

class Sin(Operation):
    
    a = Input(types.Wildcard(), description="Input value")

    def operation(self):
        return "sin",

class Cos(Operation):
        
    a = Input(types.Wildcard(), description="Input value")

    def operation(self):
        return "cos",

class Tan(Operation):
            
    a = Input(types.Wildcard(), description="Input value")

    def operation(self):
        return "tan",

class ArcSin(Operation):
    a = Input(types.Wildcard(), description="Input value")

    def operation(self):
        return "asin",

class ArcCos(Operation):

    a = Input(types.Wildcard(), description="Input value")

    def operation(self):
        return "acos",

class ArcTan(Operation):
    
    a = Input(types.Wildcard(), description="Input value")

    def operation(self):
        return "atan",

# Logical operations

class LogicalAnd(Operation):
    
    a = Input(types.Wildcard(), description="First operand")
    b = Input(types.Wildcard(), description="Second operand")

    def operation(self):
        return "logical_and",

Node.register_operation(NodeOperation.LOGICAL_AND, LogicalAnd, types.Wildcard(), types.Wildcard())

class LogicalOr(Operation):

    a = Input(types.Wildcard(), description="First operand")
    b = Input(types.Wildcard(), description="Second operand")

    def operation(self):
        return "logical_or",

Node.register_operation(NodeOperation.LOGICAL_OR, LogicalOr, types.Wildcard(), types.Wildcard())

class LogicalNot(Operation):

    a = Input(types.Wildcard(), description="First operand")

    def operation(self):
        return "logical_not",

Node.register_operation(NodeOperation.LOGICAL_NOT, LogicalNot, types.Wildcard())

class Stack(Operation):
    """Merges three single channel images into three channel image.
    """

    inputs = List(Input(types.Wildcard(mindim=1)), description="Two or more input tensors")

    def _init(self):
        if len(self.inputs) == 0:
            raise NodeException("No inputs provided", node=self)

    def input_values(self):
        return [self.inputs[int(name)] for name, _ in enumerate(self.inputs)]

    def get_inputs(self):
        return [(str(k), types.Wildcard(mindim=1)) for k, _ in enumerate(self.inputs)]

    def duplicate(self, _origin=None, **inputs):
        config = self.dump()
        for k, v in inputs.items():
            i = int(k)
            assert i >= 0 and i < len(config["inputs"])
            config["inputs"][i] = v
        return self.__class__(_origin=_origin, **config)

    def operation(self):
        return "stack",
