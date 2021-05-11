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

    def test_make_point(self):
        with GraphBuilder() as graph:
            Output(outputs=[MakePoint(1, 1)])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        np.testing.assert_equal(sample[0], np.array([1, 1]))

    def test_make_points(self):
        gt = np.array([[1, 1], [2, 2], [3, 4]])
        with GraphBuilder() as graph:
            Output(outputs=[MakePoints(gt.flatten().tolist())])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        np.testing.assert_equal(sample[0], gt)

    def test_points_arithmetic(self):
        gt = np.array([[1, 1], [2, 2], [1, 1]])
        with GraphBuilder() as graph:
            points = MakePoints(gt.flatten().tolist())
            point = MakePoint(2, 2)

            a = (((points + points) - 3) * point) / 2

            Output(outputs=[a])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        gt = (((gt + gt) - 3) * 2) / 2

        np.testing.assert_equal(sample[0], gt)

    def test_make_rectangle(self):

        with GraphBuilder() as graph:
            r = MakeRectangle(0, 0, 10, 10)
            Output(outputs=[r])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        np.testing.assert_equal(sample[0], np.array([0, 0, 10, 10]))

    def test_resize_rectangle(self):

        with GraphBuilder() as graph:
            r = MakeRectangle(0, 10, 10, 20)
            Output(outputs=[ResizeRectangle(r, 2)])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        np.testing.assert_equal(sample[0], np.array([[-5, 5, 15, 25]]).T)

    def test_make_view(self):
        with GraphBuilder() as graph:
            n1 = AffineView(x=0, y=1)
            n2 = RotateView(angle=3)
            Output(outputs=[Chain(inputs=[n1, n2])])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        # TODO: calculate transform manually, compare matrices
        #self.assertEqual(pipeline.run(1)[0], 3)


    def test_make_rectangle(self):
        with GraphBuilder() as graph:
            Output(outputs=[MakeRectangle(0, 0, 10, 10)])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)
        np.testing.assert_equal(sample[0], np.array([[0], [10], [0], [10]]))