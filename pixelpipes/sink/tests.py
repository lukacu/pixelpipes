

import unittest

import numpy as np

from ..compiler import Compiler
from ..graph import GraphBuilder, Output, Constant
from ..list import ConstantList
from . import PipelineDataLoader

class TestSinks(unittest.TestCase):

    def test_numpy_sink(self):
        
        with GraphBuilder() as builder:
            Output(outputs=[Constant(1), ConstantList([10, 20, 30])])
        graph = builder.build()

        batch_size = 10

        loader = PipelineDataLoader(graph, batch_size, 1)

        for batch in loader:
            self.assertIsInstance(batch[0], np.ndarray)
            self.assertIsInstance(batch[1], np.ndarray)
            self.assertEqual(batch[0].shape, (batch_size, 1))
            self.assertEqual(batch[1].shape, (batch_size, 3, 1))

            return

    def test_torch_list(self):
        # TODO: fix this test
        return

        import torch
        from pixelpipes.engine import FloatList

        with GraphBuilder() as builder:
            n1 = ListSource(FloatList([0.5, 0.1, 0.3]))
            Output(outputs=[n1])

        compiler = Compiler()
        pipeline = compiler.compile(builder)

        sample = pipeline.run_torch(1)

        self.assertIsInstance(sample[0], torch.Tensor)
        self.assertEqual(sample[0][0], 0.5)
