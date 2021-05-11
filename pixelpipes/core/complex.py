
from attributee import String

from .. import types
from ..core import Copy
from ..node import Macro, Input, Reference
from ..graph import GraphBuilder

class GetElement(Macro):
    """Extract field from a complex structure type

    This macro exposes only selected field of an input structure as an output,
    enabling processing of that data.

    Inputs:
     - source: Input resource
     - field: Name of the structure field

    Category: complex
    """

    source = Input(types.Complex())
    element = String()

    def validate(self, **inputs):
        super().validate(**inputs)

        if not self.element in inputs["source"].elements:
            raise ValueError("Element {} not in structure".format(self.element))

        return inputs["source"].elements[self.element]

    def expand(self, inputs, parent: Reference):

        with GraphBuilder(prefix=parent) as builder:
            
            Copy(source=inputs["source"].type.access(self.element, inputs["source"]), _name=parent)

            return builder.nodes()