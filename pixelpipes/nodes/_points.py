
from attributee import String, Float, Integer, Map, List, Boolean, Number

from pixelpipes import Node, Input
import pixelpipes.engine as engine
import pixelpipes.types as types

class BoundingBox(Node):

    points = Input(types.Points())

    def _output(self) -> types.Type:
        return types.BoundingBox()

    def operation(self):
        return engine.BoundingBox()

class ViewPoints(Node):

    source = Input(types.Points())
    view = Input(types.View())

    def operation(self):
        return engine.ViewPoints()

    def validate(self, **inputs):
        super().validate(**inputs)

        source_type = inputs["source"]
        
        return types.Points(source_type.length)

