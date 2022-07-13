
import unittest

import numpy as np

from .points import PointsBounds, PointsCenter, PointsFromRectangle
from ..graph import Operation, Input, Macro, Graph, outputs
from ..list import GetElement

from .. import types

class MakeRectangle(Operation): 
    """
    Creates a bounding box from four values.
    """

    x1 = Input(types.Integer(), description="Left")
    y1 = Input(types.Integer(), description="Top")
    x2 = Input(types.Integer(), description="Right")
    y2 = Input(types.Integer(), description="Bottom")

    def operation(self):
        return "make_rectangle",

    def infer(self, **inputs):
        return types.Rectangle()

class ResizeRectangle(Macro):
    """
    Scales existing rectangle by a factor.
    """

    source = Input(types.Rectangle())
    factor = Input(types.Float())

    def expand(self, source, factor):
        points = PointsFromRectangle(source)
        center = PointsCenter(points)
        points = ((points - center) * factor) + center
        return PointsBounds(points)

class RectangleArea(Macro):
    """
    Calculates and area under rectangle.
    """

    source = Input(types.Rectangle())

    def expand(self, source):
        x1 = GetElement(source, 0)
        y1 = GetElement(source, 1)
        x2 = GetElement(source, 2)
        y2 = GetElement(source, 3)
        return ((x2 - x1) * (y2 - y1))

class Tests(unittest.TestCase):    

    def test_make_rectangle(self):
        from ..compiler import Compiler

        with Graph() as graph:
            outputs(MakeRectangle(0, 0, 10, 10))

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_equal(output[0], np.array([0, 0, 10, 10]))    

    def test_rectangle_resize(self):
        from ..compiler import Compiler

        with Graph() as graph:
            r = MakeRectangle(0, 10, 10, 20)
            outputs(ResizeRectangle(r, 2))

        pipeline = Compiler(debug=True).build(graph)
        output = pipeline.run(1)

        np.testing.assert_equal(output[0], np.array([-5, 5, 15, 25]))

    def test_rectangle_area(self):
        from ..compiler import Compiler

        with Graph() as graph:
            r = MakeRectangle(0, 10, 10, 20)
            outputs(RectangleArea(r))

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_equal(output[0], 100)