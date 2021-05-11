

from attributee import String, Number

from .. import types
from ..node import Node, Input

class CompilerException(Exception):
    pass

class Variable(Node):

    node_description = "Variable placeholder that can be overriden later"

    name = String()
    default = Number()

class Conditional(Node):
    """Conditional selection
    
    Node that executes conditional selection, output of branch "true" will be selected if
    the "condition" is not zero, otherwise output of branch "false" will be selected.

    Inputs:
     * true (Primitve): Use this data if condition is true
     * false (Primitve): Use this data if condition is false
     * condition (Integer): Condition to test

    Parameters

    Category: flow
    """

    true = Input(types.Primitive())
    false = Input(types.Primitive())
    condition = Input(types.Integer())

    def operation(self):
        return "_condition", [[False]]

    def validate(self, **inputs):
        super().validate(**inputs)
        return inputs["true"].common(inputs["false"])

from .core import Compiler