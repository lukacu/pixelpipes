import numbers

from attributee import List, Number, Enumeration

from pixelpipes import Node, Input, hidden, ComparisonOperation, NodeException
import pixelpipes.engine as engine
import pixelpipes.types as types

def _ensure_node(value):
    if isinstance(value, _NumericNode):
        return value
    elif isinstance(value, numbers.Number):
        return Constant(value=value)
    else:
        raise ValueError("Value is not a numeric node")

@hidden
class _NumericNode(Node):

    def __add__(self, other):
        other = _ensure_node(other)
        return Add(a=self, b=other)

    def __sub__(self, other):
        other = _ensure_node(other)
        return Subtract(a=self, b=other)

    def __mul__(self, other):
        other = _ensure_node(other)
        return Multiply(a=self, b=other)

    def __truediv__(self, other):
        other = _ensure_node(other)
        return Divide(a=self, b=other)

    def __pow__(self, other):
        other = _ensure_node(other)
        return Power(a=self, b=other)

    def __mod__(self, other):
        other = _ensure_node(other)
        return Modulo(a=self, b=other)

    def __neg__(self):
        return Multiply(a=self, b=int(-1))

class Constant(_NumericNode):

    node_name = "Constant"
    node_description = "Outputs a constant number"
    node_category = "numeric"

    value = Number(default=0)

    def operation(self):
        return engine.Constant(self.value)

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

class UniformDistribution(_NumericNode):

    node_name = "Uniform distribution"
    node_description = "Samples values between min and max"
    node_category = "numeric"

    min = Input(types.Float())
    max = Input(types.Float())

    def _output(self) -> types.Type:
        return types.Float()

    def operation(self):
        return engine.UniformDistribution()

class NormalDistribution(_NumericNode):

    node_name = "Normal distribution"
    node_description = "Samples values between from normal distribution"
    node_category = "numeric"

    mean = Input(types.Float(), default=0)
    sigma = Input(types.Float(), default=1)

    def _output(self) -> types.Type:
        return types.Float()

    def operation(self):
        return engine.NormalDistribution()

class Round(_NumericNode):

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
        return engine.Round()

@hidden
class _BinaryOperator(_NumericNode):

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
        return engine.Add()

    def _compute(self, a, b):
        return Constant.resolve_type(a + b)

class Multiply(_BinaryOperator):

    def operation(self):
        return engine.Multiply()

    def _compute(self, a, b):
        return Constant.resolve_type(a * b)

class Subtract(_BinaryOperator):

    def operation(self):
        return engine.Subtract()

    def _compute(self, a, b):
        return Constant.resolve_type(a - b)

class Divide(_BinaryOperator):

    def operation(self):
        return engine.Divide()

    def _compute(self, a, b):
        return types.Float(a / b)

class Power(_BinaryOperator):

    def operation(self):
        return engine.Power()

    def _compute(self, a, b):
        return Constant.resolve_type(a ** b)

class Modulo(_BinaryOperator):

    a = Input(types.Integer())
    b = Input(types.Integer())

    def operation(self):
        return engine.Modulo()

    def _compute(self, a, b):
        return types.Integer(a % b)

class Maximum(_BinaryOperator):

    def operation(self):
        return engine.Maximum()

    def _compute(self, a, b):
        return Constant.resolve_type(max(a, b))

class Minimum(_BinaryOperator):

    def operation(self):
        return engine.Minimum()

    def _compute(self, a, b):
        return Constant.resolve_type(min(a, b))

class Threshold(Node):

    threshold = Number()
    comparison = Enumeration(ComparisonOperation)
    source = Input(types.Number())

    def _output(self):
        return types.Integer()

    def operation(self):
        return engine.Threshold(self.threshold, self.comparison)

class _ThresholdsComparison(Node):

    thresholds = List(Number())
    comparison = List(Enumeration(ComparisonOperation))
    inputs = List(Input(types.Number()))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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