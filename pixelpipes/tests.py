
import unittest

import numpy as np

from pixelpipes import *
from pixelpipes.resources import *

class TestPipes(unittest.TestCase):

    def test_numerical(self):
        builder = GraphBuilder()
        n1 = builder.add(Constant(value=5))
        n2 = builder.add(Constant(value=15))
        n3 = builder.add(Expression(source="((x ^ 2 - y) * 2) / 5 + 2", variables=dict(x=n1, y=n2)))

        builder.add(Output(outputs=[n3]))
        compiler = Compiler()
        graph = builder.build()
        pipeline = compiler.compile(graph)

        self.assertEqual(pipeline.run(1)[0], 2)


    def test_view(self):
        builder = GraphBuilder()
        n1 = builder.add(AffineView(x=0, y=1))
        n2 = builder.add(RotateView(view=n1, angle=3))

        builder.add(Output(outputs=[n2]))
        compiler = Compiler()
        graph = builder.build()
        pipeline = compiler.compile(graph)

        pipeline.run(1)

        # TODO: calculate transform manually, compare matrices
        #self.assertEqual(pipeline.run(1)[0], 3)

    def test_numpy_out(self):
        builder = GraphBuilder()
        n1 = builder.add(Constant(value=5))
        n2 = builder.add(IdentityView())

        builder.add(Output(outputs=[n1, n2]))
        compiler = Compiler()
        graph = builder.build()
        pipeline = compiler.compile(graph)

        sample = pipeline.run(1, engine.Convert.NUMPY)

        self.assertIsInstance(sample[0], np.ndarray)
        self.assertEqual(sample[0][0], 5)
        self.assertIsInstance(sample[1], np.ndarray)

