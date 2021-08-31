

from attributee import List
from pixelpipes.core import Copy
from .points import PointsBounds, PointsCenter, PointsFromRectangle
from ..graph import Graph, GraphBuilder
from ..core.list import ListElement

from .. import types
from .types import Rectangle
from ..node import Node, Input, Macro

class MakeRectangle(Node): 
    """Make Rectangle
    
    Creates a bounding box from four values.

    Inputs:
        - x1: left bound
        - x2: right bound
        - y1: top bound
        - y2: bottom bound

    Category: Rectangle
    """

    x1 = Input(types.Integer())
    x2 = Input(types.Integer())
    y1 = Input(types.Integer())
    y2 = Input(types.Integer())

    def operation(self):
        return "geometry:make_rectangle",

    def _output(self):
        return Rectangle()

class ResizeRectangle(Macro):
    """Resize Rectangle
    
    Scales existing rectangle by a factor.

    Inputs:
        - source: A rectangle type
        - factor: Scaling factor

    Category: Rectangle
    """

    source = Input(Rectangle())
    factor = Input(types.Float())

    def _output(self) -> types.Type:
        return Rectangle()

    def expand(self, inputs, parent: str):
        
        with GraphBuilder(prefix=parent) as builder:
            points = PointsFromRectangle(inputs["source"])
            center = PointsCenter(points)

            points = ((points - center) * inputs["factor"]) + center

            PointsBounds(points, _name=parent)

        return builder.nodes()

class RectangleArea(Macro):
    """Rectangle Area
    
    Calculates and area under rectangle.

    Inputs:
        - source: A rectangle type

    Category: Rectangle
    """

    source = Input(Rectangle())

    def _output(self) -> types.Type:
        return types.Float()

    def expand(self, inputs, parent: str):
        
        with GraphBuilder(prefix=parent) as builder:
            x1 = ListElement(inputs["source"], 0)
            y1 = ListElement(inputs["source"], 1)
            x2 = ListElement(inputs["source"], 2)
            y2 = ListElement(inputs["source"], 3)

            
            Copy((x2 - x1) * (y2 - y1), _name = parent )

        return builder.nodes()