
from pixelpipes import Node, Input, Macro
from pixelpipes.nodes.list import ListBuild
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

class PointsCenter(Node):
    """Points center
    
    Computes center of point set as an average of all coordinates

    Inputs:
        source: A list of points

    Category: points
    """

    source = Input(types.Points())

    def operation(self):
        return engine.PointsCenter()

    def _output(self) -> types.Type:
        return types.Point()

class MakePoint(Node):
    """Make point
    
    Creates a point from two numerical inputs

    Inputs:
     - x: X coordinate
     - y: Y coordinate

    Category: points
    """

    x = Input(types.Number())
    y = Input(types.Number())

    def operation(self):
        return engine.PointFromInputs()

    def _output(self) -> types.Type:
        return types.Point()

class MakeBounds(Macro):
    """Make bounding box
    
    Creates a bounding box from four values.

    Inputs:
     - left: left bound
     - top: top bound
     - right: right bound
     - bottom: bottom bound

    Category: points
    """

    left = Input(types.Number())
    top = Input(types.Number())
    right = Input(types.Number())
    bottom = Input(types.Number())

    def _output(self) -> types.Type:
        return types.BoundingBox()

    def expand(self, inputs, parent: str):
        return ListBuild([inputs["left"][0], inputs["top"][0], inputs["right"][0], inputs["bottom"][0]])

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