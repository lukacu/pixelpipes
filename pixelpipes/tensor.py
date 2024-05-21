
from attributee import List

from . import types, DataType
from .graph import Macro, Operation, Input, NodeException, EnumerationInput


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

class Reshape(Operation):
    """Reshapes input tensor to desired shape.
    """

    source = Input(types.Wildcard(), description="Input tensor")
    shape = Input(types.IntegerList(), description="Shape of the output tensor")

    def operation(self):
        return "reshape",
    
class Transpose(Operation):
    
    source = Input(types.Wildcard(), description="Input tensor")
    axes = Input(types.IntegerList(), description="Permutation of the dimensions")

    def operation(self):
        return "transpose",

class Convert(Operation):
    """Converts input to different primitive data type.
    """

    source = Input(types.Wildcard(), description="Input value")
    dtype = EnumerationInput(DataType, default="Integer", description="Desired data type")

    def operation(self):
        return "convert",

class Float(Macro):
    """Converts input to float. A utility macro for Convert operation.
    """

    source = Input(types.Wildcard(), description="Input value")

    def expand(self, source):
        return Convert(source, "Float")

class Integer(Macro):
    """Converts input to integer. A utility macro for Convert operation.
    """

    source = Input(types.Wildcard(), description="Input value")

    def expand(self, source):
        return Convert(source, "Integer")
    
class Boolean(Macro):
    """Converts input to boolean. A utility macro for Convert operation.

    """
    source = Input(types.Wildcard(), description="Input value")

    def expand(self, source):
        return Convert(source, "Boolean")
    
