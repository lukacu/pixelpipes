

from attributee import List

from ..graph import NodeOperation, Input, Macro, Operation, ValidationException, Node

from .. import types
from ..list import ListBuild
from ..types import Point, Points, Rectangle, View

PointsBroadcastType = types.Union(types.Float(), Point(), Points())
PointBroadcastType = types.Union(types.Float(), Point())

def _infer_type_point(source1: types.Data, source2: types.Data):

    point1 = Point().common(source1)
    point2 = Point().common(source2)

    if (isinstance(point1, types.Anything) and isinstance(point1, types.Anything)):
        raise types.TypeException("At least one input should be a point")

    return Point()

class _ArithmeticPointOperation(Operation):

    a = Input(PointBroadcastType)
    b = Input(PointBroadcastType)

    def infer(self, **inputs):
        return _infer_type_points(inputs["a"], inputs["b"])

class AddPoint(_ArithmeticPointOperation):
    def operation(self):
        return "point2d_add", 

class SubtractPoint(_ArithmeticPointOperation):
    def operation(self):
        return "point2d_subtract",

class MultiplyPoint(_ArithmeticPointOperation):
    def operation(self):
        return "point2d_multiply",

class DividePoint(_ArithmeticPointOperation):
    def operation(self):
        return "point2d_divide", 

class _ArithmeticPointsOperation(Operation):

    a = Input(PointsBroadcastType)
    b = Input(PointsBroadcastType)

    def infer(self, a, b):
        return a.common(b)
        
class AddPoints(_ArithmeticPointsOperation):
    def operation(self):
        return "points2d_add",

class SubtractPoints(_ArithmeticPointsOperation):
    def operation(self):
        return "points2d_subtract",

class MultiplyPoints(_ArithmeticPointsOperation):
    def operation(self):
        return "points2d_multiply",

class DividePoints(_ArithmeticPointsOperation):
    def operation(self):
        return "points2d_divide",

class PointsBounds(Operation):
    """Computes an axis aligned bounging box on a set of points
    """

    points = Input(Points(), description="List of points")

    def infer(self, points) -> types.Data:
        return Rectangle()

    def operation(self):
        return "bounding_box",

# TODO: move to view.py
class ViewPoints(Operation):
    """Transforms points with a given view.

    Inputs:
        - source: A list of points
        - view: View type

    Category: points
    """

    source = Input(Points(), description="Input list of points")
    view = Input(View(), description="Transform")

    def operation(self):
        return "opencv:view_points",

    def infer(self, source, view):
        return Points(source[0])

class PointsCenter(Operation):
    """Computes center of point set as an average of all coordinates"""

    source = Input(Points(), description="Source points")

    def operation(self):
        return "points2d_center",

    def infer(self, source) -> types.Data:
        return Point()

class MakePoint(Operation):
    """Creates a point from two numerical inputs
    """
    x = Input(types.Float(), description="X coordinate")
    y = Input(types.Float(), description="Y coordinate")

    def operation(self):
        return "make_point2d",

    def infer(self, x, y) -> types.Data:
        return Point()

class MakePoints(Operation):
    """Creates a list of points from an even number of numerical inputs
    """

    inputs = List(Input(types.Float()), description="Input values, should be an even number of them")

    def operation(self):
        return "make_points2d",

    def infer(self, **inputs):
        if len(self.inputs) % 2 == 1:
            raise ValidationException("Even number of inputs expected")
        return Points(length=int(len(self.inputs) / 2))

    def input_values(self):
        return [self.inputs[int(name)] for name, _ in self.get_inputs()]

    def get_inputs(self):
        return [(str(k), types.Float()) for k, _ in enumerate(self.inputs)]

    def duplicate(self, _origin=None, **inputs):
        config = self.dump()
        for k, v in inputs.items():
            i = int(k)
            assert i >= 0 and i < len(config["inputs"])
            config["inputs"][i] = v
        return self.__class__(_origin=_origin, **config)


class MakeRectangle(Macro):
    """Creates a bounding box from four values.
    """

    left = Input(types.Float(), description="Left value")
    top = Input(types.Float(), description="Top value")
    right = Input(types.Float(), description="Right value")
    bottom = Input(types.Float(), description="Bottom value")

    def expand(self, left, top, right, bottom):
        return ListBuild([left, top, right, bottom])

class PointsFromRectangle(Operation):
    """Convert bounding box to a list of points
    """

    source = Input(Rectangle(), description="Input rectangle")

    def operation(self):
        return "points_from_rectangle",

    def infer(self, source) -> types.Data:
        return Points(length=4)
