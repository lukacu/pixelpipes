

import unittest

import numpy as np

from ..compiler import Compiler
from ..graph import Graph, Constant, outputs
from . import PipelineDataLoader

class TestSinks(unittest.TestCase):

    def test_sink_PipelineDataLoader(self):
        
        with Graph() as graph:
            outputs(Constant(1), Constant([10, 20, 30]))

        batch_size = 10

        loader = PipelineDataLoader(Compiler().build(graph), batch_size, 1)

        for batch in loader:
            self.assertIsInstance(batch[0], np.ndarray)
            self.assertIsInstance(batch[1], np.ndarray)
            self.assertEqual(batch[0].shape, (batch_size, 1))
            self.assertEqual(batch[1].shape, (batch_size, 3))
            return

    """
    # TODO FIX
    def test_torch_list(self):

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
    """
