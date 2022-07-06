import unittest
import numpy as np

from ..graph import outputs
from ..geometry.points import MakePoint, MakePoints
from ..geometry.view import AffineView, RotateView,Chain
from .rectangle import MakeRectangle, ResizeRectangle
from ..graph import Graph
from ..compiler import Compiler

class TestPoints(unittest.TestCase):    
    
    def test_make_points(self):
        from ..compiler import Compiler

        gt = np.array([[1, 1], [2, 2], [3, 4]])

        with Graph() as graph:
            outputs(MakePoint(1, 1))
            outputs(MakePoints(gt.flatten().tolist()))

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_equal(output[0], np.array([[1, 1]]))
        np.testing.assert_equal(output[1], gt)

    def test_arithmetic(self):
        from ..compiler import Compiler

        gt = np.array([[1, 1], [2, 2], [1, 1]])

        with Graph() as graph:
            points = MakePoints(gt.flatten().tolist())
            point = MakePoint(2, 2)
            a = (((points + points) - 3) * point) / 2
            outputs(a)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        gt = (((gt + gt) - 3) * 2) / 2
        np.testing.assert_equal(output[0], gt)

    def test_serialization(self):

        from ..graph import compare_serialized

        gt = np.array([[1, 1], [2, 2], [1, 1]])
        with Graph() as graph:
            points = MakePoints(gt.flatten().tolist())
            point = MakePoint(2, 2)
            a = (((points + points) - 3) * point) / 5
            outputs(a)

        compare_serialized(graph)

class TestRectangle(unittest.TestCase):    

    def test_make_rectangle(self):

        with Graph() as graph:
            outputs(MakeRectangle(0, 0, 10, 10))

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_equal(output[0], np.array([0, 0, 10, 10]))    

    def test_rectangle_resize(self):

        with Graph() as graph:
            r = MakeRectangle(0, 10, 10, 20)
            outputs(ResizeRectangle(r, 2))

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_equal(output[0], np.array([-5, 5, 15, 25]))

    def test_views(self):

        with Graph() as graph:
            n1 = AffineView(x=0, y=1)
            n2 = RotateView(angle=3)
            outputs(Chain(inputs=[n1, n2]))

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        # TODO: calculate transform manually, compare matrices
        #np.testing.assert_equal(output[0], np.array([0]))

