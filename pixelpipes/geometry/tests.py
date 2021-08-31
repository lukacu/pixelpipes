import unittest
import numpy as np
from pixelpipes.resource import MakeResource

from ..core import Output
from ..geometry.points import MakePoint, MakePoints
from ..geometry.view import AffineView, RotateView,Chain
from .rectangle import MakeRectangle, ResizeRectangle
from ..graph import GraphBuilder
from ..compiler import Compiler

class TestPipes(unittest.TestCase):    

    def test_geometry_points_MakePoint(self):

        with GraphBuilder() as graph:
            Output(outputs=[MakePoint(1, 1)])

        pipeline = Compiler().compile(graph)
        output = pipeline.run(1)

        np.testing.assert_equal(output[0], np.array([1, 1]))

    def test_geometry_points_MakePoints(self):

        gt = np.array([[1, 1], [2, 2], [3, 4]])

        with GraphBuilder() as graph:
            Output(outputs=[MakePoints(gt.flatten().tolist())])

        pipeline = Compiler().compile(graph)
        output = pipeline.run(1)

        np.testing.assert_equal(output[0], gt)

    def test_geometry_points_arithmetic_operations(self):

        gt = np.array([[1, 1], [2, 2], [1, 1]])

        with GraphBuilder() as graph:
            points = MakePoints(gt.flatten().tolist())
            point = MakePoint(2, 2)
            a = (((points + points) - 3) * point) / 2
            Output(outputs=[a])

        pipeline = Compiler().compile(graph)
        output = pipeline.run(1)

        gt = (((gt + gt) - 3) * 2) / 2
        np.testing.assert_equal(output[0], gt)

    def test_geometry_MakeRectangle(self):

        with GraphBuilder() as graph:
            Output(outputs=[MakeRectangle(0, 0, 10, 10)])

        pipeline = Compiler().compile(graph)
        output = pipeline.run(1)

        np.testing.assert_equal(output[0], np.array([[0], [10], [0], [10]]))    

    def test_geometry_rectangle_ResizeRectangle(self):

        with GraphBuilder() as graph:
            r = MakeRectangle(0, 10, 10, 20)
            Output(outputs=[ResizeRectangle(r, 2)])

        pipeline = Compiler().compile(graph)
        output = pipeline.run(1)

        np.testing.assert_equal(output[0], np.array([[-5, 5, 15, 25]]).T)

    def test_geometry_view_View(self):

        with GraphBuilder() as graph:
            n1 = AffineView(x=0, y=1)
            n2 = RotateView(angle=3)
            Output(outputs=[Chain(inputs=[n1, n2])])

        pipeline = Compiler().compile(graph)
        output = pipeline.run(1)

        # TODO: calculate transform manually, compare matrices
        np.testing.assert_equal(output[0], np.array([0]))
