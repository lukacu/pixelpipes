

from attributee import List

from ..graph import Input, Macro, Operation, SeedInput

from .. import types
from ..list import MakeList
from ..types import Point, Points, Rectangle, View

PointsBroadcastType = types.Union(types.Float(), Point(), Points())
PointBroadcastType = types.Union(types.Float(), Point())

class PointsBounds(Operation):
    """Computes an axis aligned bounging box on a set of points
    """

    points = Input(Points(), description="List of points")

    def operation(self):
        return "bounding_box",

class PointsCenter(Operation):
    """Computes center of point set as an average of all coordinates"""

    source = Input(Points(), description="Source points")

    def operation(self):
        return "points2d_center",

class MakePoint(Operation):
    """Creates a point from two numerical inputs
    """
    x = Input(types.Float(), description="X coordinate")
    y = Input(types.Float(), description="Y coordinate")

    def operation(self):
        return "make_point2d",


class MakePoints(Operation):
    """Creates a list of points from an even number of numerical inputs
    """

    inputs = List(Input(types.Float()), description="Input values, should be an even number of them")

    def operation(self):
        return "make_points2d",

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
        return MakeList([left, top, right, bottom])

class PointsFromRectangle(Operation):
    """Convert bounding box to a list of points
    """

    source = Input(Rectangle(), description="Input rectangle")

    def operation(self):
        return "points_from_rectangle",

class RandomPoints(Operation):
    """Generates a list of random points
    """

    count = Input(types.Integer(), description="Number of points")
    seed = SeedInput()

    def operation(self):
        return "random_points2d",