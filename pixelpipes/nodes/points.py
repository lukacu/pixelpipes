
from attributee import String, Float, Integer, Map, List, Boolean, Number

from pixelpipes import Node, Input
import pixelpipes.engine as engine
import pixelpipes.types as types

class BoundingBox(Node):

    node_name = "Bounging box"
    node_description = "Computes an axis aligned bounging box on a set of points"
    node_category = "points"

    points = Input(types.Points())

    def _output(self) -> types.Type:
        return types.BoundingBox()

    def operation(self):
        return engine.BoundingBox()

class ViewPoints(Node):

    node_name = "View points"
    node_description = "Transforms points with a given view"
    node_category = "points"

    source = Input(types.Points())
    view = Input(types.View())

    def operation(self):
        return engine.ViewPoints()

    def validate(self, **inputs):
        super().validate(**inputs)

        source_type = inputs["source"]
        
        return types.Points(source_type.length)

class PointsFromBoundingBox(Node):
    """Convert bounding box to points

    Inputs:
        source: A bounding box
    """

    source = Input(types.BoundingBox())

    def operation(self):
        return engine.PointsFromBoundingBox()

    def _output(self) -> types.Type:
        return types.Points(length=4)