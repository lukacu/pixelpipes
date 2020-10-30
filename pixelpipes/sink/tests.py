

import unittest

import numpy as np

from pixelpipes import *
from pixelpipes.resources import *

class TestSinks(unittest.TestCase):

    def test_torch(self):
        if not engine.torch:
            return

        import torch

        builder = GraphBuilder()
        n1 = builder.add(Constant(value=5))
        n2 = builder.add(IdentityView())

        builder.add(Output(outputs=[n1, n2]))
        compiler = Compiler()
        graph = builder.build()
        pipeline = compiler.compile(graph)

        sample = pipeline.run(1, engine.Convert.TORCH)

        self.assertIsInstance(sample[0], torch.Tensor)
        self.assertEqual(sample[0][0], 5)
        self.assertIsInstance(sample[1], torch.Tensor)
