import unittest
import numpy as np

from ..graph import outputs
from ..geometry.points import MakePoint, MakePoints
from ..geometry.view import AffineView, RotateView, Chain, IdentityView, ViewPoints
from .rectangle import MakeRectangle, ResizeRectangle
from ..graph import Graph
from ..compiler import Compiler

from pixelpipes.tests import TestBase

class TestPoints(TestBase):    
    
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

        from ..tests import compare_serialized

        gt = np.array([[1, 1], [2, 2], [1, 1]])
        with Graph() as graph:
            points = MakePoints(gt.flatten().tolist())
            point = MakePoint(2, 2)
            a = (((points + points) - 3) * point) / 5
            outputs(a)

        compare_serialized(graph)

    def test_points_view(self):

        with Graph() as graph:
            n0 = IdentityView()
            n1 = AffineView(x=0, y=1)
            n2 = RotateView(angle=np.pi)
            p = MakePoints([0, 0, 1, 1, 2, 2])
            outputs(ViewPoints(p, Chain(inputs=[n0, n1, n2])))

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_almost_equal(output[0], np.array([[0, 1], [-1, 0], [-2, -1]]))

    def test_view_calculation(self):
        from ..geometry.view import FocusView, CenterView


        with Graph() as graph:
            rectangle = MakeRectangle(0, 0, 10, 10)
            v1 = FocusView(rectangle, 1)
            v2 = CenterView(rectangle)
            outputs(v1, v2)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        self.compare_arrays(output[0], np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]], dtype=np.float32))
        self.compare_arrays(output[1], np.array([[1, 0, -5], [0, 1, -5], [0, 0, 1]], dtype=np.float32))
  
class TestRectangle(TestBase):    

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

        # TODO: calculate transform manually (translate, rotate), compare matrices

        reference = np.array([[0, 1], [1, 0], [0, 1], [1, 0]]) 


                #np.testing.assert_equal(output[0], np.array([0]))

