
from typing import Optional
from attributee import List
from attributee.primitives import Enumeration

from ..graph import BinaryOperation, Node, Input, Macro, ValidationException

from .. import ArithmeticOperations, types
from ..list import ListBuild
from .types import Point, Points, Rectangle, View

PointsBroadcastType = types.Union(types.Float(), Point(), Points())
PointBroadcastType = types.Union(types.Float(), Point())

def _infer_type_point(source1: types.Type, source2: types.Type):

    point1 = Point().common(source1)
    point2 = Point().common(source2)

    if (isinstance(point1, types.Any) and isinstance(point1, types.Any)):
        raise types.TypeException("At least one input should be a point")

    return Point()
class _ArithmeticPointOperation(Node):

    a = Input(PointBroadcastType)
    b = Input(PointBroadcastType)

    def validate(self, **inputs):
        super().validate(**inputs)
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

for op, cb in [(BinaryOperation.ADD, AddPoint), (BinaryOperation.SUBTRACT, SubtractPoint), (BinaryOperation.MULIPLY, MultiplyPoint), (BinaryOperation.DIVIDE, DividePoint)]:
    Node.register_operation(op, cb, _infer_type_point, Point(), PointBroadcastType)
    Node.register_operation(op, cb, _infer_type_point, PointBroadcastType, Point())

def _infer_type_points(source1: types.Type, source2: types.Type):

    list1 = Points().common(source1)
    list2 = Points().common(source2)

    if (isinstance(list1, types.Any) and isinstance(list2, types.Any)):
        raise types.TypeException("At least one input should be a list of points")

    if (not isinstance(list1, types.Any) and not isinstance(list2, types.Any)):
        return list1.common(list2)

    return Points()
class _ArithmeticPointsOperation(Node):

    a = Input(PointsBroadcastType)
    b = Input(PointsBroadcastType)

    def validate(self, **inputs):
        super().validate(**inputs)
        return _infer_type_points(inputs["a"], inputs["b"])
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

for op, cb in [(BinaryOperation.ADD, AddPoints), (BinaryOperation.SUBTRACT, SubtractPoints), (BinaryOperation.MULIPLY, MultiplyPoints), (BinaryOperation.DIVIDE, DividePoints)]:
    Node.register_operation(op, cb, _infer_type_points, Points(), PointsBroadcastType)
    Node.register_operation(op, cb, _infer_type_points, PointsBroadcastType, Points())

class PointsBounds(Node):
    """Points Bounds

    Computes an axis aligned bounging box on a set of points

    Inputs:
        - points: A list of points

    Category: points
    """

    points = Input(Points())

    def _output(self) -> types.Type:
        return Rectangle()

    def operation(self):
        return "bounding_box",

# TODO: move to view.py
class ViewPoints(Node):
    """View Points
    
    Transforms points with a given view.

    Inputs:
        - source: A list of points
        - view: View type

    Category: points
    """

    source = Input(Points())
    view = Input(View())

    def operation(self):
        return "opencv:view_points",

    def validate(self, **inputs):
        super().validate(**inputs)

        source_type = inputs["source"]
        
        return Points(source_type.length)

class PointsCenter(Node):
    """Points center
    
    Computes center of point set as an average of all coordinates

    Inputs:
        - source: A list of points

    Category: points
    """

    source = Input(Points())

    def operation(self):
        return "points2d_center",

    def _output(self) -> types.Type:
        return Point()

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
        return "make_point2d",

    def _output(self) -> types.Type:
        return Point()

class MakePoints(Node):
    """Make points
    
    Creates a list of points from an even number of numerical inputs

    Inputs:
        - x: X coordinate
        - y: Y coordinate

    Category: points
    """

    inputs = List(Input(types.Number()))

    def operation(self):
        return "make_points2d",

    def validate(self, **inputs):
        super().validate(**inputs)

        if len(self.inputs) % 2 == 1:
            raise ValidationException("Even number of inputs expected")

        return Points(length=int(len(self.inputs) / 2))


    def input_values(self):
        return [self.inputs[int(name)] for name, _ in self.get_inputs()]

    def get_inputs(self):
        return [(str(k), types.Number()) for k, _ in enumerate(self.inputs)]

    def duplicate(self, _origin=None, **inputs):
        config = self.dump()
        for k, v in inputs.items():
            i = int(k)
            assert i >= 0 and i < len(config["inputs"])
            config["inputs"][i] = v
        return self.__class__(_origin=_origin, **config)


class MakeRectangle(Macro):
    """Make bounding box rectangle
    
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
        return Rectangle()

    def operation(self):
        return "bounding_box",

    def expand(self, inputs, parent: str):
        return ListBuild([inputs["left"], inputs["top"], inputs["right"], inputs["bottom"]])

class PointsFromRectangle(Node):
    """Convert bounding box to points

    Inputs:
        - source: A bounding box

    Category: points
    """

    source = Input(Rectangle())

    def operation(self):
        return "points_from_rectangle",

    def _output(self) -> types.Type:
        return Points(length=4)