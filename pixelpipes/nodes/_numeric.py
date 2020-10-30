from attributee import String, Float, Integer, Map, List, Boolean, Number

from pixelpipes import Node, Input
import pixelpipes.engine as engine
import pixelpipes.types as types


class UniformDistribution(Node):

    min = Input(types.Float())
    max = Input(types.Float())

    def _output(self) -> types.Type:
        return types.Float()

    def operation(self):
        return engine.UniformDistribution()

class NormalDistribution(Node):

    mean = Input(types.Float(), default=0)
    sigma = Input(types.Float(), default=1)

    def _output(self) -> types.Type:
        return types.Float()

    def operation(self):
        return engine.NormalDistribution()

class Round(Node):

    source = Input(types.Float())

    def _output(self) -> types.Type:
        return types.Integer()

    def operation(self):
        return engine.Round()

class Associative(Node):

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

class Subtract(Node):

    a = Input(types.Number())
    b = Input(types.Number())

    def operation(self):
        return engine.Subtract()

    def validate(self, **inputs):
        super().validate(**inputs)

        integer = isinstance(inputs["a"], types.Integer) and isinstance(inputs["b"], types.Integer)

        return types.Integer() if integer else types.Float()


class Divide(Node):

    a = Input(types.Number())
    b = Input(types.Number())

    def operation(self):
        return engine.Divide()

    def _output(self):
        return types.Float()

class Power(Node):

    a = Input(types.Number())
    b = Input(types.Number())

    def operation(self):
        return engine.Power()

    def _output(self):
        return types.Float()