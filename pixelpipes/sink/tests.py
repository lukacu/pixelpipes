

import unittest

import numpy as np

from pixelpipes import GraphBuilder, Compiler, Output, Constant, with_torch, types
from pixelpipes.nodes import IdentityView, ListSource

class TestSinks(unittest.TestCase):

    def test_torch(self):
        if not with_torch():
            return

        import torch

        with GraphBuilder() as builder:
            n1 = Constant(value=5)
            n2 = IdentityView()
            Output(outputs=[n1, n2])

        compiler = Compiler()
        graph = builder.build()
        pipeline = compiler.compile(graph)

        sample = pipeline.run_torch(1)

        self.assertIsInstance(sample[0], torch.Tensor)
        self.assertEqual(sample[0][0], 5)
        self.assertIsInstance(sample[1], torch.Tensor)

    def test_torch_list(self):
        if not with_torch():
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
