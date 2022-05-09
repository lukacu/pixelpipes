import unittest
import numpy as np

from .. import write_pipeline, read_pipeline
from ..graph import Output, outputs
from ..geometry.points import MakePoint, MakePoints
from ..geometry.view import AffineView, RotateView,Chain
from .rectangle import MakeRectangle, ResizeRectangle
from ..graph import GraphBuilder
from ..compiler import Compiler

class TestPipes(unittest.TestCase):    

    def test_geometry_points_MakePoint(self):

        with GraphBuilder() as graph:
            outputs(MakePoint(1, 1))

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_equal(output[0], np.array([1, 1]))

    def test_geometry_points_MakePoints(self):

        gt = np.array([[1, 1], [2, 2], [3, 4]])

        with GraphBuilder() as graph:
            outputs(MakePoints(gt.flatten().tolist()))

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_equal(output[0], gt)

    def test_geometry_points_arithmetic_operations(self):

        gt = np.array([[1, 1], [2, 2], [1, 1]])

        with GraphBuilder() as graph:
            points = MakePoints(gt.flatten().tolist())
            point = MakePoint(2, 2)
            a = (((points + points) - 3) * point) / 2
            outputs(a)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        gt = (((gt + gt) - 3) * 2) / 2
        np.testing.assert_equal(output[0], gt)

    def test_geometry_MakeRectangle(self):

        with GraphBuilder() as graph:
            outputs(MakeRectangle(0, 0, 10, 10))

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_equal(output[0], np.array([[0], [10], [0], [10]]))    

    def test_geometry_rectangle_ResizeRectangle(self):

        with GraphBuilder() as graph:
            r = MakeRectangle(0, 10, 10, 20)
            outputs(ResizeRectangle(r, 2))

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_equal(output[0], np.array([[-5, 5, 15, 25]]).T)

    def test_geometry_view_View(self):

        with GraphBuilder() as graph:
            n1 = AffineView(x=0, y=1)
            n2 = RotateView(angle=3)
            outputs(Chain(inputs=[n1, n2]))

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        # TODO: calculate transform manually, compare matrices
        #np.testing.assert_equal(output[0], np.array([0]))

    def test_make_rectangle(self):
        with GraphBuilder() as graph:
            outputs(MakeRectangle(0, 0, 10, 10))

        pipeline = Compiler().build(graph)
        sample = pipeline.run(1)
        np.testing.assert_equal(sample[0], np.array([[0], [10], [0], [10]]))

    def test_serialization(self):

        import tempfile

        gt = np.array([[1, 1], [2, 2], [1, 1]])
        with GraphBuilder() as graph:
            points = MakePoints(gt.flatten().tolist())
            point = MakePoint(2, 2)
            a = (((points + points) - 3) * point) / 5
            outputs(a)

        compiler = Compiler()

        pipeline1 = compiler.build(graph)

        tmp = tempfile.mktemp()

        write_pipeline(tmp, compiler.compile(graph))

        pipeline2 = read_pipeline(tmp)

        a = pipeline1.run(1)
        b = pipeline2.run(1)
        np.testing.assert_equal(a[0], b[0])
