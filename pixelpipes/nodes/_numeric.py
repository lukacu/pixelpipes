import numbers

from attributee import List, Number

from pixelpipes import Node, Input
import pixelpipes.engine as engine
import pixelpipes.types as types

def _ensure_node(value):
    if isinstance(value, NumericNode):
        return value
    elif isinstance(value, numbers.Number):
        return Constant(value=value)
    else:
        raise ValueError("Value is not a numeric node")

class NumericNode(Node):

    def __add__(self, other):
        other = _ensure_node(other)
        return Add(inputs=[self, other])

    def __sub__(self, other):
        other = _ensure_node(other)
        return Subtract(a=self, b=other)

    def __mul__(self, other):
        other = _ensure_node(other)
        return Multiply(inputs=[self, other])

    def __truediv__(self, other):
        other = _ensure_node(other)
        return Divide(a=self, b=other)

    def __pow__(self, other):
        other = _ensure_node(other)
        return Power(a=self, b=other)

    def __neg__(self):
        return Multiply(inputs=[self, -1])

class Constant(NumericNode):

    value = Number()

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

class UniformDistribution(NumericNode):

    min = Input(types.Float())
    max = Input(types.Float())

    def _output(self) -> types.Type:
        return types.Float()

    def operation(self):
        return engine.UniformDistribution()

class NormalDistribution(NumericNode):

    mean = Input(types.Float(), default=0)
    sigma = Input(types.Float(), default=1)

    def _output(self) -> types.Type:
        return types.Float()

    def operation(self):
        return engine.NormalDistribution()

class Round(NumericNode):

    source = Input(types.Float())

    def _output(self) -> types.Type:
        return types.Integer()

    def operation(self):
        return engine.Round()

class Associative(NumericNode):

    inputs = List(Input(types.Number()))

    def validate(self, **inputs):
        super().validate(**inputs)

        integer = True
        for i in inputs.values():
            integer &= isinstance(i, types.Integer)

        return types.Integer() if integer else types.Float()

    def input_values(self):
        return [self.inputs[int(name)] for name, _ in self._gather_inputs()]

    def _gather_inputs(self):
        return [(str(k), types.Float()) for k, _ in enumerate(self.inputs)]

    def duplicate(self, **inputs):
        config = self.dump()
        for k, v in inputs.items():
            i = int(k)
            assert i >= 0 and i < len(config["inputs"])
            config["inputs"][i] = v
        return self.__class__(**config)

class Add(Associative):

    def operation(self):
        return engine.Add()

class Multiply(Associative):

    def operation(self):
        return engine.Multiply()

class Subtract(NumericNode):

    a = Input(types.Number())
    b = Input(types.Number())

    def operation(self):
        return engine.Subtract()

    def validate(self, **inputs):
        super().validate(**inputs)

        integer = isinstance(inputs["a"], types.Integer) and isinstance(inputs["b"], types.Integer)

        return types.Integer() if integer else types.Float()


class Divide(NumericNode):

    a = Input(types.Number())
    b = Input(types.Number())

    def operation(self):
        return engine.Divide()

    def _output(self):
        return types.Float()

class Power(NumericNode):

    a = Input(types.Number())
    b = Input(types.Number())

    def operation(self):
        return engine.Power()

    def _output(self):
        return types.Float()