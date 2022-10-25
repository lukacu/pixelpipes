

import unittest

import numpy as np

from ..compiler import Compiler
from ..graph import Graph, Constant, outputs
from . import PipelineDataLoader

def _run_sink_text(constructor, batch_size, workers):

    with Graph() as graph:
        outputs(Constant(1), Constant([10, 20, 30]))

    batch_size = 10

    loader = constructor(Compiler().build(graph), batch=batch_size, workers=workers)

    for _ in zip(loader, range(1000)):
        pass

class TestSinks(unittest.TestCase):

    def test_sink_single_worker(self):
        
        _run_sink_text(PipelineDataLoader, 32, 1)

    def test_sink_multi_worker(self):
        
        _run_sink_text(PipelineDataLoader, 32, 50)

    def test_sink_pytorch(self):

        try:
            from . import TorchDataLoader
            _run_sink_text(TorchDataLoader, 32, 1)
        except ImportError:
            return

        

  
